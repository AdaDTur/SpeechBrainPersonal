from google.colab import drive
drive.mount('/content/gdrive')

!pip install speechbrain

!pip install sentencepiece transformers --quiet

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np


outputfile = open('outputfile.txt', 'w')
for file in os.listdir('/content/gdrive/MyDrive/dev-other'):
  filename = os.path.join('/content/gdrive/MyDrive/dev-other', file)
  for item1 in os.listdir(filename):
    filename = os.path.join(filename, item1)
    if os.path.isdir(filename):
      for item2 in os.listdir(filename):
        if(item2[-5:] == '.flac'):
          file = os.path.join(filename, item2)
          print(file)
          asr_out = str(*map(list, zip(* asr_model.transcribe_file(file, InferenceDetailLevel.TOP_K_HYP_SCORES) )))
          outputfile.write(file + '\t' + asr_out + '\n')

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
  predsHash[id] = myList[0]
  splitPreds = myList[0].split()
  for word in splitPreds:
    if word not in wordDict:
      wordDict[word] = tokenCount
      tokenCount += 1

import collections
from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch, accumulatable_wer_stats

import collections
from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch, accumulatable_wer_stats
refs = []
hyps = []
batches = []
count = 0
for key,value in predsHash.items():
  #print(key, value)
  #print(transcriptsHash[key])
  ref = [wordDict[word] for word in transcriptsHash[key].split()]
  hyp = [wordDict[word] for word in value.split()]
  refs.append(ref)
  hyps.append(hyp)
  #print(refs, hyps)
  batches.append([refs, hyps])


stats = collections.Counter()
for batch in batches:
  refs, hyps = batch
  stats = accumulatable_wer_stats(refs, hyps, stats)
  #print(refs, '\n', hyps)
  #print(stats)
print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))

model_name = 'bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())

preds = open('/content/gdrive/MyDrive/outputfile2.txt')
log = open('/content/gdrive/MyDrive/logfile.txt', 'w')
import sys
import ast
count = 0
hyps = []
batches = []
refs = []
predsHash = {}
lines = preds.readlines()
for line in lines:
  words = line.split()
  pred = ' '.join(words[1:])
  id = words[0].split('.')[0]
  myList = ast.literal_eval(pred)
  min = sys.maxsize
  print(id)
  log.write(id +'\n')
  for line in myList:
    words = line.split()
    for word in words:
      if word not in wordDict:
        wordDict[word] = tokenCount
        tokenCount += 1
    line = line.lower()
    log.write(line +'\n')
    val = score(sentence=line, model=model, tokenizer=tokenizer)
    if val < min:
      best_hyp = line
      min = val
  ref = [wordDict[word] for word in transcriptsHash[id].split()]
  log.write(str(ref) + '\n')
  refs.append(ref)
  #print(ref)
  log.write(best_hyp + str(min) + '\n')
  best_hyp = best_hyp.upper()
  hyp = [wordDict[word] for word in best_hyp.split()]
  hyps.append(hyp)
  batches.append([refs, hyps])


stats = collections.Counter()
for batch in batches:
  refs, hyps = batch
  stats = accumulatable_wer_stats(refs, hyps, stats)
  #print(refs, '\n', hyps)
  #print(stats)
print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))
