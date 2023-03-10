from google.colab import drive
drive.mount('/content/gdrive')

!pip install sentencepiece transformers --quiet

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np

model_name = 'bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def score(model, tokenizer, sentence):
    #print("sentence", sentence)
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    tokens = tokenizer.tokenize(sentence)
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
        #print("loss", loss)
    return loss.item()

%%capture
!git clone https://github.com/speechbrain/speechbrain/
%cd /content/speechbrain/
!pip install -r requirements.txt
!git checkout ctc-prefix-beamsearch
!pip install -e .

%%capture
!pip install transformers numba

import hashlib
import subprocess
from speechbrain.pretrained.fetching import fetch


def fetch_patch_hparams(source, 
                        patch: str, 
                        classname="CustomInterface", 
                        savedir=None, 
                        use_auth_token=False,
                        hparams_file="hyperparams.yaml"):
  
    # download hparams file as usual
    if savedir is None:
        savedir = f"./pretrained_models/{classname}-{hashlib.md5(source.encode('UTF-8', errors='replace')).hexdigest()}"
    hparams_local_path = fetch(hparams_file, source=source, savedir=savedir, use_auth_token=use_auth_token, overwrite=True)

    # put the patch file in the non-symlink files
    patch_path = str(hparams_local_path) + ".patch"
    with open(patch_path, 'w') as f:
      f.write(patch)
    
    # apply patch on the cached/sym-link:ed yaml file
    cmd_out = subprocess.run(["patch", hparams_local_path.resolve(), patch_path], capture_output=True)
    print("Args: {}\n\nstdout: {}\nstderr: {}\n\nPatch code: {} (0: ok)\n".format(cmd_out.args, cmd_out.stdout.decode("utf-8") , cmd_out.stderr.decode("utf-8"), cmd_out.returncode ))

# Replaces decoder & introduces new ScoreBuilder; enables top-5 hypotheses, also
patch_yaml_crdnn_rnnlm = """--- hyperparams.yaml
+++ hyperparams.yaml
@@ -124,11 +124,27 @@
     normalize: !ref <normalizer>
     model: !ref <enc>
 
-decoder: !new:speechbrain.decoders.S2SRNNBeamSearchLM
+# Scorer
+topk: 5  # multiple hypotheses
+
+coverage_scorer: !new:speechbrain.decoders.scorer.CoverageScorer
+    vocab_size: !ref <output_neurons>
+
+rnnlm_scorer: !new:speechbrain.decoders.scorer.RNNLMScorer
+    language_model: !ref <lm_model>
+    temperature: !ref <temperature_lm>
+
+scorer: !new:speechbrain.decoders.scorer.ScorerBuilder
+    full_scorers: [!ref <rnnlm_scorer>,
+                   !ref <coverage_scorer>]
+    weights:
+       rnnlm: !ref <lm_weight>
+       coverage: !ref <coverage_penalty>
+
+decoder: !new:speechbrain.decoders.S2SRNNBeamSearcher
     embedding: !ref <emb>
     decoder: !ref <dec>
     linear: !ref <seq_lin>
-    language_model: !ref <lm_model>
     bos_index: !ref <bos_index>
     eos_index: !ref <eos_index>
     min_decode_ratio: !ref <min_decode_ratio>
@@ -137,11 +153,9 @@
     eos_threshold: !ref <eos_threshold>
     using_max_attn_shift: !ref <using_max_attn_shift>
     max_attn_shift: !ref <max_attn_shift>
-    coverage_penalty: !ref <coverage_penalty>
-    lm_weight: !ref <lm_weight>
     temperature: !ref <temperature>
-    temperature_lm: !ref <temperature_lm>
-
+    topk: !ref <topk>
+    scorer: !ref <scorer>
 
 modules:
     normalizer: !ref <normalizer>

"""

fetch_patch_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", patch=patch_yaml_crdnn_rnnlm, classname="EncoderDecoderASR")

from speechbrain.pretrained import EncoderDecoderASR


asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav')

from speechbrain.pretrained.interfaces_topk import EncoderDecoderASRTopK, InferenceDetailLevel

# fetch_patch_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", patch=patch_yaml_crdnn_rnnlm, classname="EncoderDecoderASRTopK")
asr_model = EncoderDecoderASRTopK.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

import os
print( asr_model.transcribe_file('/speechbrain/example.wav') )

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

for key,value in predsHash.items():
  ref = [wordDict[word] for word in transcriptsHash[key].split()]
  hyp = [wordDict[word] for word in value.split()]
  refs.append(ref)
  hyps.append(hyp)

stats = collections.Counter()
for batch in batches:
    refs, hyps = batch
    stats = accumulatable_wer_stats(refs, hyps, stats)
    print(refs, '\n', hyps)
    print(stats)
print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))

from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch, accumulatable_wer_stats
import collections
refs = []
hyps = []
batches = []
count = 0
for key,value in predsHash.items():
    ref = [wordDict[word] for word in transcriptsHash[key].split()]
    hyp = [wordDict[word] for word in value.split()]
    refs.append(ref)
    hyps.append(hyp)
    batches.append([refs, hyps])


stats = collections.Counter()
for batch in batches:
  refs, hyps = batch
  stats = accumulatable_wer_stats(refs, hyps, stats)


print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))

import collections
from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch, accumulatable_wer_stats

preds = open('/content/gdrive/MyDrive/outputfile2.txt')
log = open('/content/gdrive/MyDrive/logfile3.txt', 'w')
import sys
import ast
count = 0
counter = 0
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
  minTok = []
  for line in myList:
    words = line.split()
    for word in words:
      if word not in wordDict:
        wordDict[word] = tokenCount
        tokenCount += 1
    tokens = tokenizer.tokenize(line)
    minTok.append(len(tokens))
  for line in myList:
    words = line.split()
    for word in words:
      if word not in wordDict:
        wordDict[word] = tokenCount
        tokenCount += 1
    tokens = tokenizer.tokenize(line)
    line = line.lower()
    log.write(line +'\n')
    val = score(sentence=line, model=model, tokenizer=tokenizer)
    val = val * (len(tokens) - (np.min(minTok)/2 + 1))
    if val < min:
      best_hyp = line
      min = val
  ref = [wordDict[word] for word in transcriptsHash[id].split()]
  log.write(str(ref) + '\n')
  refs.append(ref)
  log.write(best_hyp + str(min) + '\n')
  best_hyp = best_hyp.upper()
  hyp = [wordDict[word] for word in best_hyp.split()]
  hyps.append(hyp)
  batches.append([refs, hyps])


stats = collections.Counter()
for batch in batches:
  refs, hyps = batch
  stats = accumulatable_wer_stats(refs, hyps, stats)
#print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))

print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))

preds = open('/content/gdrive/MyDrive/outputfile2.txt')
log = open('/content/gdrive/MyDrive/logfile4.txt', 'w')
import sys
import ast
count = 0
counter = 0
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
  log.write(id +'\n')
  for line in myList:
    best_hyp = line
    break
  ref = [wordDict[word] for word in transcriptsHash[id].split()]
  log.write(str(ref) + '\n')
  refs.append(ref)
  #print(best_hyp, '\n', transcriptsHash[id])
  #log.write(best_hyp + str(min) + '\n')
  best_hyp = best_hyp.upper()
  hyp = [wordDict[word] for word in best_hyp.split()]
  hyps.append(hyp)
  batches.append([refs, hyps])

stats = collections.Counter()
for batch in batches:
  refs, hyps = batch
  stats = accumulatable_wer_stats(refs, hyps, stats)
print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))
