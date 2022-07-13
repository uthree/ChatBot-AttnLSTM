import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FeedForwardNetwork(nn.Module):
    """Some Information about FeedForwardNetwork"""
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    """Some Information about EncoderLayer"""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, reverse=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FeedForwardNetwork(d_model, d_ff, d_model)

    def forward(self, x):
        res1 = x
        x = self.norm1(x)
        x, _ = self.mha(x, x, x)
        x += res1
        res2 = x
        x = self.norm2(x)
        x = self.ffn(x) + res2
        return x

class Encoder(nn.Module):
    """Stack of encoder layers"""
    def __init__(self, num_layers, bidirectional=True, **kwargs):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(EncoderLayer(**kwargs))

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

class DecoderLayer(nn.Module):
    """Some Information about DecoderLayer"""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(DecoderLayer, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model, bidirectional=False, batch_first=True)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FeedForwardNetwork(d_model, d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, tgt, memory, state=None):
        res1 = tgt
        tgt, _ = self.mha(self.norm1(tgt), memory, memory)
        tgt_in = tgt + res1
        res2 = tgt_in
        if state == None:
            tgt_out, state = self.lstm(self.norm2(tgt_in))
        else:
            tgt_out, state = self.lstm(self.norm2(tgt_in), state)
        tgt_out = tgt_out + res2

        res3 = tgt_out
        tgt_out = self.ffn(tgt_out) + res3
        return tgt_out, state


class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self, num_layers, **kwargs):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DecoderLayer(**kwargs))

    def forward(self, tgt, memory, state):
        out_states = []
        for i in range(len(self.layers)):
            if state[i] == None:
                tgt, o_s = self.layers[i](tgt, memory)
            else:
                tgt, o_s = self.layers[i](tgt, memory, state[i])
            out_states.append(o_s)
        return tgt, out_states

class Seq2Seq(nn.Module):
    """Some Information about Seq2Seq"""
    def __init__(self, num_encoder_layers=6, num_decoder_layers=6, bidirectional_encoder=True, d_model=512, dim_ffn=1024, n_heads=8, vocab_size=30000):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.hid2vocab = nn.Linear(d_model, vocab_size)
        self.encoder = Encoder(num_encoder_layers, bidirectional=bidirectional_encoder, d_model=d_model, n_heads=n_heads, d_ff=dim_ffn)
        self.decoder = Decoder(num_decoder_layers, d_model=d_model, n_heads=n_heads, d_ff=dim_ffn)
        self.num_decoder_layers = num_decoder_layers

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        mem = self.encoder(src)
        state = [None] * self.num_decoder_layers
        tgt, _ = self.decoder(tgt, mem, state)
        tgt = self.hid2vocab(tgt)
        return tgt
