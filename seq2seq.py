import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FeedForwardNetwork(nn.Module):
    """Some Information about FeedForwardNetwork"""
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    """Some Information about EncoderLayer"""
    def __init__(self, d_model=512, n_heads=8, d_ff=1024, dropout=0.1, reverse=False):
        super(EncoderLayer, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model, bidirectional=False, batch_first=True)
        self.ffn = FeedForwardNetwork(d_model*2, d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.reverse = reverse
    def forward(self, src):
        if self.reverse:
            src = torch.flip(src, [1])
        src_out, state = self.lstm(src)
        src = self.layernorm(self.dropout(src_out)) + src # [batch_size, seq_len, d_model]
        attn_v, attn_weight = self.attn(src, src, src)
        attn_v = torch.cat([src, attn_v], dim=-1)
        attn_v = self.dropout(attn_v) 
        ffn_out = self.ffn(attn_v) # [batch_size, seq_len, d_model]
        src = self.layernorm(src + ffn_out)
        if self.reverse:
            src = torch.flip(src, [1])
        return src, state

class Encoder(nn.Module):
    """Stack of encoder layers"""
    def __init__(self, num_layers, bidirectional=True, **kwargs):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0 or bidirectional == False:
                self.layers.append(EncoderLayer(**kwargs, reverse=False))
            else:
                self.layers.append(EncoderLayer(**kwargs, reverse=True))

    def forward(self, src):
        out_states = []
        for layer in self.layers:
            src, state = layer(src)
            out_states.append(state)
        return src, out_states

class DecoderLayer(nn.Module):
    """Some Information about DecoderLayer"""
    def __init__(self, d_model=512, n_heads=8, d_ff=1024, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model, bidirectional=False, batch_first=True)
        self.ffn = FeedForwardNetwork(d_model*2, d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self, tgt, memory, state):
        tgt_out, state = self.lstm(tgt, state)
        tgt = self.layernorm(self.dropout(tgt_out) + tgt) # [batch_size, seq_len, d_model]
        attn_v, attn_weight = self.attn(tgt, memory, memory)
        attn_v = torch.cat([tgt, attn_v], dim=-1)
        attn_v = self.dropout(attn_v)
        ffn_out = self.ffn(attn_v) # [batch_size, seq_len, d_model]
        tgt = self.layernorm(tgt + ffn_out)
        return tgt, state

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
            h, c = state[i]
            i_s = (h, c)
            tgt, o_s = self.layers[i](tgt, memory, i_s)
            out_states.append(o_s)
        return tgt, out_states

class Seq2Seq(nn.Module):
    """Some Information about Seq2Seq"""
    def __init__(self, num_encoder_layers=4, num_decoder_layers=4, bidirectional_encoder=True, d_model=512, dim_ffn=1024, n_heads=8, dropout=0.1, vocab_size=30000):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.hid2vocab = nn.Linear(d_model, vocab_size)
        self.encoder = Encoder(num_encoder_layers, bidirectional=bidirectional_encoder, d_model=d_model, n_heads=n_heads, d_ff=dim_ffn, dropout=dropout)
        self.decoder = Decoder(num_decoder_layers, d_model=d_model, n_heads=n_heads, d_ff=dim_ffn, dropout=dropout)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src, enc_state = self.encoder(src)
        tgt, _ = self.decoder(tgt, src, enc_state)
        tgt = self.hid2vocab(tgt)
        return tgt

