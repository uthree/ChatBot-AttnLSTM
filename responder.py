import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from seq2seq import Seq2Seq

class Responder:
    def __init__(self, sentencepiece_model_path='./sentencepiece.model', padding_idx=3, **kwargs):
        self.s2s = Seq2Seq(**kwargs)
        self.spm = spm.SentencePieceProcessor()
        self.spm.Load(sentencepiece_model_path)
        self.spm.SetEncodeExtraOptions("bos:eos")
        self.paddng_idx = padding_idx
    
    def train(self, dataset, num_epoch=1, batch_size=1, device=torch.device('cpu')):
        self.s2s.train()
        self.s2s.to(device)
        optimizer = optim.Adam(self.s2s.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index=self.paddng_idx)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        bar = tqdm(total=len(dataset) * num_epoch)
        for epoch in range(num_epoch):
            for i, (src, tgt) in enumerate(dataloader):
                src = src.to(device)
                tgt = tgt.to(device)
                optimizer.zero_grad()
                output = self.s2s(src, tgt[:, :-1])
                loss = criterion(torch.flatten(output, start_dim=0, end_dim=1), torch.flatten(tgt[:, 1:], start_dim=0, end_dim=1))
                loss.backward()
                optimizer.step()
                bar.set_description(f'Epoch: {epoch}/{num_epoch}, Loss: {loss.item():.4f}')
                bar.update(batch_size)
                if i % 3000 == 0:
                    torch.save(self.s2s, './seq2seq.pt')
            torch.save(self.s2s, './seq2seq.pt')
        bar.close()

    def predict_tensor(self, src, device=torch.device('cpu'), max_output_len = 64):
        self.s2s.eval()
        self.s2s.to(device)
        src = src.to(device)
        tgt = torch.zeros(src.shape[0], 1).long().to(device)
        memory, state = self.s2s.encoder(self.s2s.embedding(src))
        for i in range(max_output_len):
            output, state = self.s2s.decoder(self.s2s.embedding(tgt[:, -1:]), memory, state)
            output_argmax = self.s2s.hid2vocab(output).argmax(dim=-1)
            #print(output_argmax)
            tgt = torch.cat((tgt, output_argmax), dim=1)
        return tgt

    def padding_after_eos(self, ids, eos_idx=2, padding_idx=3):
        eos = False
        res = []
        for id in ids:
            if id == eos_idx:
                eos = True
            if eos:
                res.append(padding_idx)
            else:
                res.append(id)
        return res

    def predict_sentences(self, sentences, device=torch.device('cpu'), max_output_len = 64):
        src = [self.spm.EncodeAsIds(sentence) for sentence in sentences]
        src = torch.tensor(src).long()
        tgt = self.predict_tensor(src, device, max_output_len)
        tgt = [self.padding_after_eos(ids) for ids in tgt.tolist()]
        return [self.spm.DecodeIdsWithCheck(tgt[i]) for i in range(len(sentences))]

    @classmethod
    def load(cls, model_path, spm_path):
        resp = Responder()
        resp.s2s = torch.load(model_path)
        resp.spm = spm.SentencePieceProcessor()
        resp.spm.Load(spm_path)
        resp.spm.SetEncodeExtraOptions("bos:eos")
        return resp

    def save(self, model_path):
        torch.save(self.s2s, model_path)

