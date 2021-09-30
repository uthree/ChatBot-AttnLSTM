import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sentencepiece as spm

class DialogueDataset(torch.utils.data.Dataset):
    """Some Information about DialogueDataset"""
    def __init__(self, data_file_pathes, sentencepiece_model_path, max_len=32, padding_idx=3):
        super(DialogueDataset, self).__init__()
        # load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sentencepiece_model_path)
        self.sp.SetEncodeExtraOptions("bos:eos")

        # load data
        law_data = []
        for data_file_path in data_file_pathes:
            with open(data_file_path, 'r') as f:
                for line in f:
                    law_data.append(line.strip())
        
        # tokenize data
        self.data = []
        for line in law_data:
            ids = self.sp.EncodeAsIds(line)
            # padding
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [padding_idx] * (max_len - len(ids))
            self.data.append(ids)
        
        self.data = torch.LongTensor(self.data)

    def __getitem__(self, index):
        return self.data[index], self.data[index + 1]

    def __len__(self):
        return len(self.data) - 1