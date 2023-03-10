from google.colab import drive
drive.mount('/content/gdrive')

!pip install sentencepiece transformers --quiet

from transformers import BloomConfig, BloomModel, AutoTokenizer, BloomForCausalLM
import torch
import numpy as np

"""The function below calculates average probability of each token in the sentence given all the other tokens. 

In masked language models, this does not amount to the total probability of the whole sentence (the conditional probabilities do not cancel each other out), but it is still a useful measure of a "naturallness" of a sentence (for more details read e.g. https://arxiv.org/abs/1910.14659). 
"""

configuration = BloomConfig()
model = BloomModel(configuration)
configuration = model.config

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-7b1")

def score(sentence):
  inputs = tokenizer(sentence, return_tensors="pt")
  outputs = model(**inputs, labels=inputs["input_ids"])
  loss=outputs.loss
  return np.exp(loss.cpu().detach().numpy())

texts = ['therefore from the first they would have been imperfect in bobbiny\'s size and power', 'therefore from the first they would have been imperfect in bobbiny size and power', 'therefore from the first they would have been imperfect in bodily size and power', 'therefore from the first they would have been imperfect in botany size and power', 'therefore from the first they would have been imperfect in bobby\'s size and power' ]
sentence = "therefore from the first they would have been imperfect in bobbiny\'s size and power"
for text in texts:
  print(score(text), text)

%%capture
!git clone https://github.com/speechbrain/speechbrain/
%cd /content/speechbrain/
!pip install -r requirements.txt
!git checkout ctc-prefix-beamsearch
!pip install -e .

import collections
from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch, accumulatable_wer_stats

transcripts = open('/content/gdrive/MyDrive/librispeech.txt')
tokenCount = 0
wordDict = {}
transcriptsHash = {}
lines = transcripts.readlines()
for line in lines:
  words = line.split()
  id = words[0]
  transcript = ' '.join(words[1:]) 
  transcriptsHash[id] = transcript
  splitTrans = transcript.split()
  for word in splitTrans:
    if word not in wordDict:
      wordDict[word] = tokenCount
      tokenCount += 1

preds = open('/content/gdrive/MyDrive/outputfile2.txt')
import ast
predsHash = {}
lines = preds.readlines()
for line in lines:
  words = line.split()
  id = words[0].split('.')[0]
  pred = ' '.join(words[1:])
  myList = ast.literal_eval(pred)
  for pred in myList:
    splitPreds = pred.split()
    for word in splitPreds:
      if word not in wordDict:
        wordDict[word] = tokenCount
        tokenCount += 1

preds = open('/content/gdrive/MyDrive/outputfile2.txt')
log = open('/content/gdrive/MyDrive/logfile5.txt', 'w')
import sys
import ast
counter = 0
hyps = []
batches = []
refs = []
predsHash = {}
counter = 0
lines = preds.readlines()
for line in lines:
  words = line.split()
  pred = ' '.join(words[1:])
  id = words[0].split('.')[0]
  myList = ast.literal_eval(pred)
  print(id)
  log.write(id +'\n')
  min = 9999999999
  first_hyp = ""
  for line in myList:
    line = line.lower()
    if first_hyp == "":
      first_hyp = line.upper()
    val = score(sentence=line)
    if val < min:
      min = val
      best_hyp = line
  ref = [wordDict[word] for word in transcriptsHash[id].split()]
  refs.append(ref)
  best_hyp = best_hyp.upper()
  log.write(best_hyp + '\n')
  #print(best_hyp)
  hyp = [wordDict[word] for word in best_hyp.split()]
  hyps.append(hyp)
  batches.append([refs, hyps])
  if counter == 100:
    break
  counter += 1

stats = collections.Counter()
for batch in batches:
  refs, hyps = batch
  stats = accumulatable_wer_stats(refs, hyps, stats)
print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))
