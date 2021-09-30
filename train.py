from torch._C import device
from dialogue_dataset import DialogueDataset
from responder import Responder
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ds = DialogueDataset(["F:/nucc/nucc.txt", "F:/logs/nucc_discord_supecial_tokens.txt"], "./sentencepiece.model")
if os.path.exists("./seq2seq.pt"):
    print("Loading model...")
    responder = Responder.load('seq2seq.pt', './sentencepiece.model')
else:
    print("Building model...")
    responder = Responder(sentencepiece_model_path="./sentencepiece.model", vocab_size=10000)
responder.train(ds, device=torch.device("cuda"), batch_size=200, num_epoch=100)