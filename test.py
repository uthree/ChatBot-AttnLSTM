from dialogue_dataset import DialogueDataset
from responder import Responder
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

responder = Responder.load('seq2seq.pt', './sentencepiece.model')
join_past_logs = 1
logs = []
while True:
    user_utterance = input('USER > ')
    logs.append(user_utterance)
    bot_responce = responder.predict_sentences(['<sep>'.join(logs[-join_past_logs:])], noise_gain=0.3)[0]
    bot_responce = bot_responce.replace('<sep>', '\n')
    for resp in bot_responce.split('\n'):
        print('BOT > ' + resp)
        #time.sleep(1)
    logs.append(bot_responce)
