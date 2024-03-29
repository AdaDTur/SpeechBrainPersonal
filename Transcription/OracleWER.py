from google.colab import drive
drive.mount('/content/gdrive')

!pip install sentencepiece transformers --quiet

# %%capture
# # For pip installation -> install Brian's PR
# !git clone https://github.com/speechbrain/speechbrain/
# %cd /content/speechbrain/
# !pip install -r requirements.txt
# !git checkout ctc-prefix-beamsearch
# !pip install -e .

# %%capture
# !pip install transformers numba

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

transcripts = open('/content/gdrive/MyDrive/Librispeech/librispeech.txt')
tokenCount = 0
wordDict = {}
transcriptsHash = {}
lines = transcripts.readlines()
for line in lines:
  words = line.split()
  id = words[0]
  transcript = ' '.join(words[1:]) 
  transcriptsHash[id] = [transcript, transcript, transcript, transcript, transcript]
  splitTrans = transcript.split()
  for word in splitTrans:
    if word not in wordDict:
      wordDict[word] = tokenCount
      tokenCount += 1
print(wordDict)

preds = open('/content/gdrive/MyDrive/Librispeech/outputfile2.txt')
import ast
predsHash = {}
lines = preds.readlines()
for line in lines:
  words = line.split()
  id = words[0].split('.')[0]
  pred = ' '.join(words[1:])
  myList = ast.literal_eval(pred)
  predsHash[id] = myList
  for i in range(5):
    splitPreds = myList[i].split()
    for word in splitPreds:
      if word not in wordDict:
        wordDict[word] = tokenCount
        tokenCount += 1
print(wordDict)

from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch, accumulatable_wer_stats
import numpy as np
import collections
batches = []
count = 0
ids = [['hyp1'], ['hyp2'], ['hyp3'], ['hyp4'], ['hyp5']]
for key,value in predsHash.items():
    refs = []
    hyps = []
    for i in range(len(value)):
      ref = [wordDict[word] for word in transcriptsHash[key][i].split()]
      hyp = [wordDict[word] for word in value[i].split()]
      refs.append(ref)
      hyps.append(hyp)
    batches.append([refs, hyps])

total = 0
base_total = 0

total_tokens = 0
for batch in batches:
  refs, hyps = batch
  min = 0
  wer_details = []
  errors = []
  newrefs = [[item] for item in refs]
  newhyps = [[item] for item in hyps]
  for ids_batch, refs_batch, hyps_batch in zip(ids, newrefs, newhyps):
    details = wer_details_for_batch(ids_batch, refs_batch, hyps_batch)
    wer_details.extend(details)
  for i in range(5):
    wer = wer_details[i]['WER']
    errors.append(wer)
    if wer < wer_details[min]['WER']:
      min = i
  total += wer_details[min]['insertions'] + wer_details[min]['deletions'] + wer_details[min]['substitutions']
  base_total += wer_details[0]['insertions'] + wer_details[0]['deletions'] + wer_details[0]['substitutions']
  total_tokens += wer_details[min]['num_ref_tokens']

print(total)
print(base_total)
print(total_tokens)
print(base_total/total_tokens)
print(total/total_tokens)

ids = [['utt1'], ['utt2']]
refs = [[['a','b','c']], [['d','e']]]
hyps = [[['a','b','d']], [['d','e']]]
wer_details = []
for ids_batch, refs_batch, hyps_batch in zip(ids, refs, hyps):
    details = wer_details_for_batch(ids_batch, refs_batch, hyps_batch)
    wer_details.extend(details)
print(wer_details[1]['WER'])
