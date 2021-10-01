from dialogue_dataset import DialogueDataset
from responder import Responder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

responder = Responder.load('seq2seq.pt', './sentencepiece.model')
join_past_logs = 3
logs = []
while True:
    user_utterance = input('USER > ')
    logs.append(user_utterance)
    bot_responce = responder.predict_sentences(['<sep>'.join(logs[-join_past_logs:])])[0]
    bot_responce = bot_responce.replace('<sep>', '\n')
    print('BOT > ' + bot_responce)
    logs.append(bot_responce)
