import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dialogue_dataset import DialogueDataset
from responder import Responder

ds = DialogueDataset([ "/mnt/f/logs/nucc_discord_special_tokens.txt", "/mnt/f/nucc/nucc.txt", "/mnt/f/logs/nworkshare.txt"], "./sentencepiece.model", join_past_logs=2)
if os.path.exists("./seq2seq.pt"):
    print("Loading model...")
    responder = Responder.load('seq2seq.pt', './sentencepiece.model')
else:
    print("Building model...")
    responder = Responder(sentencepiece_model_path="./sentencepiece.model", vocab_size=10000)
responder.train(ds, device=torch.device("cuda"), batch_size=60, num_epoch=100)
