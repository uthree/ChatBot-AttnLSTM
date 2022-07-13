import os
from responder import Responder

model_path = "./seq2seq.pt"
spm_path = "./sentencepiece.model"
model = Responder.load(model_path, spm_path)

import traceback
import os
import yaml
import random

import discord
import discord.ext.commands as cmd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

COGS = []
FIRST_PERSON = ['私', '僕', 'わたし', 'ぼく', 'おれ', '俺', 'オレ']
REPLACED_FIRST_PERSON = '私'
CONJUCTION_SUFFIXES = ['けど', '、', 'したら', 'いたら', 'なら', 'だったら', 'それは', 'えっと', '...']

# Hyperparameters
gpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
max_seq_length = 64
noise_gain = 0.1
max_message_count = 1

if not gpu:
    device = torch.device('cpu')

def predict(input_sentence):
    result = model.predict_sentences([input_sentence], noise_gain=0.1)[0]

    # replace first persons
    for fp in FIRST_PERSON:
        result = result.replace(fp, REPLACED_FIRST_PERSON)
    # replace xxさん -> <USERNAME>さん
    result = result.replace(r"(.+)さん", "<name>さん")
    result = result.replace("<laugh>", "笑")
    result = result.replace("<silence>", "......")
    return result

class Bot(cmd.Bot):
    def __init__(self, prefix):
        super().__init__(command_prefix=prefix)

        for cog in COGS:
            try:
                self.load_extension(cog)
            except Exception as e:
                print(f'Failed to load extension {cog}.')
                traceback.print_exc()

    async def on_ready(self):
        print(f'Logged in as {self.user.name} ({self.user.id})')

        game = discord.Game("私にDMを送ってみてください。 / Send me direct message.")
        await self.change_presence(status=discord.Status.online, activity=game)
    
    async def on_message(self, message):
        if message.author.bot:
            return
        # check direct message channel
        if type(message.channel) == discord.DMChannel:
            for suff in CONJUCTION_SUFFIXES:
                if message.content.endswith(suff) and random.random() < 0.3:
                    return
            input_message = ''
            msgs = []
            async for msg in message.channel.history(limit=random.randint(1,max_message_count)):
                msgs.append(msg.content)
            input_message = '<sep>'.join(reversed(msgs))
            result = predict(input_message)
            result = result.replace('<name>', message.author.name)
            print(f"{input_message} -> {result}")
            result = result.replace('<sep>', '\n')
            result = result.replace('<laugh>', 'w')
            result = result.replace('<silence>', '...')
            result = result.replace('<unk>', '...')
            await message.channel.send(result)
            

default_token_data = {
    'using': 'main',
    'main': '<YOUR BOT TOKEN HERE>'
}

def main():
    if not os.path.exists("token.yml"):
        with open("token.yml", "w") as f:
            yaml.dump(default_token_data, f)
            print(
                """
                edit token.yml to add your bot token.
                in default, it will selected `main`. 
                """
            )
            exit()
    else:
        with open("token.yml", "r") as f:
            data = yaml.safe_load(f)
            token_key = data['using']
            token = data[token_key]
            
        bot = Bot(prefix=f'.')
        bot.run(token)
        
if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()
    