from transformers import AutoTokenizer, AutoModelWithLMHead #4.20.1
import pandas as pd
import transformers
import torch
from torch.nn import Dropout
from keras_preprocessing.sequence import pad_sequences
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2TokenizerFast, AdamW
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
import random

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium') #initialize tokenizer
model = AutoModelWithLMHead.from_pretrained('gpt2-medium')

train = open('/projects/traintrans2.txt', 'r')
lines = train.readlines()

maxlen = 100
tokens = [tokenizer.encode(line) for line in lines] #tokenizer converts sentences to numerical values
tokens = [token[0:maxlen] for token in tokens]
input_ids = pad_sequences(tokens, maxlen=maxlen, dtype="long", value=50256, truncating="post", padding="post")
masks = [[float(i != 50256) for i in ii] for ii in input_ids]
data = TensorDataset(torch.tensor(input_ids), torch.tensor(masks))

train_input = input_ids
train_masks = masks
train_data = TensorDataset(torch.tensor(train_input), torch.tensor(train_masks))
train_sampler = RandomSampler(train_data)

bs = 16
train_data_loader = DataLoader(train_data, sampler = train_sampler, batch_size = bs)

device = torch.device('cuda')

optimizer = AdamW(
    model.parameters(),
    lr=5e-4,
    eps=1e-8
)

total_loss = 0
num_epochs = 3

for epoch in range(num_epochs):
  itercnt = 0
  for batch in train_data_loader:
    itercnt += 1
    model.train()
    model.to(device)
    batch = tuple(t.to(device) for t in batch)
    input, mask = batch
    model.zero_grad()
    output = model(input_ids = input,
                  attention_mask = mask,
                  token_type_ids = None,
                  labels = input)
    loss = output[0]
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    if itercnt%100==0:
      print(itercnt)
      #generate()
  print(epoch,"\n")
  #generate()
  filename = "projects/gpt_train_1_"+str(epoch)+".pt"
  torch.save(model, filename)
  print("Total Loss: ",total_loss)
  total_loss = 0

