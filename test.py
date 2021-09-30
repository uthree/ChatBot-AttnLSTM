from dialogue_dataset import DialogueDataset
from responder import Responder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

responder = Responder.load('seq2seq.pt', './sentencepiece.model')
while True:
    print("BOT  >" + responder.predict_sentences([input("USER >")])[0])