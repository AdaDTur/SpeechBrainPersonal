from google.colab import drive
drive.mount('/content/gdrive')

!pip install sentencepiece transformers --quiet

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
import numpy as np

"""The function below calculates average probability of each token in the sentence given all the other tokens. 

In masked language models, this does not amount to the total probability of the whole sentence (the conditional probabilities do not cancel each other out), but it is still a useful measure of a "naturallness" of a sentence (for more details read e.g. https://arxiv.org/abs/1910.14659). 
"""

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

def score(sentence):
  tokens_tensor = tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")
  loss=model(tokens_tensor, labels=tokens_tensor)[0]
  return np.exp(loss.cpu().detach().numpy())

texts = ['therefore from the first they would have been imperfect in bobbiny\'s size and power', 'therefore from the first they would have been imperfect in bobbiny size and power', 'therefore from the first they would have been imperfect in bodily size and power', 'therefore from the first they would have been imperfect in botany size and power', 'therefore from the first they would have been imperfect in bobby\'s size and power' ]
for text in texts:
    print (text, score(text))

%%capture
!git clone https://github.com/speechbrain/speechbrain/
%cd /content/speechbrain/
!pip install -r requirements.txt
!git checkout ctc-prefix-beamsearch
!pip install -e .

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

from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch, accumulatable_wer_stats
import collections

preds = open('/content/gdrive/MyDrive/outputfile2.txt')
log = open('/content/gdrive/MyDrive/logfile4.txt', 'w')
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
  log.write(str(ref) + '\n')
  refs.append(ref)
  best_hyp = best_hyp.upper()
  hyp = [wordDict[word] for word in best_hyp.split()]
  hyps.append(hyp)
  batches.append([refs, hyps])

stats = collections.Counter()
for batch in batches:
  refs, hyps = batch
  stats = accumulatable_wer_stats(refs, hyps, stats)
print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))
