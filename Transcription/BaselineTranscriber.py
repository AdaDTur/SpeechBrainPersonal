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
    #print("tensor_input", tensor_input)
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    #print("repeat_input", repeat_input)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    #print("mask", mask)
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    #print("masked_input", masked_input)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    #print("labels", labels)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
        #print("loss", loss)
    return np.exp(loss.item())

print(score(sentence='I inquired without knowing what his aunt\'s would be it was not true.', model=model, tokenizer=tokenizer)) 
print(score(sentence='I inquired without knowing what his answer would be it was not true.', model=model, tokenizer=tokenizer))

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # For pip installation -> install Brian's PR
# !git clone https://github.com/speechbrain/speechbrain/
# %cd /content/speechbrain/
# !pip install -r requirements.txt
# !git checkout ctc-prefix-beamsearch
# !pip install -e .

# Commented out IPython magic to ensure Python compatibility.
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

from speechbrain.pretrained.interfaces_topk import EncoderDecoderASRTopK, InferenceDetailLevel

asr_model = EncoderDecoderASRTopK.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

import os
print( asr_model.transcribe_file('/speechbrain/example.wav') )

outputfile = open('/content/gdrive/MyDrive/outputfile2.txt', 'w')
for file in os.listdir('/content/gdrive/MyDrive/dev-other'):
  filename = os.path.join('/content/gdrive/MyDrive/dev-other', file)
  for item1 in os.listdir(filename):
    filename = os.path.join(filename, item1)
    if os.path.isdir(filename):
      for item2 in os.listdir(filename):
        if(item2[-5:] == '.flac'):
          file = os.path.join(filename, item2)
          print(item2)
          asr_out = str(*map(list, zip(* asr_model.transcribe_file(file, InferenceDetailLevel.TOP_K_HYP_SCORES) )))
          outputfile.write(item2 + '\t' + asr_out + '\n')

outputfile2 = open('outputfile.txt', 'r')
outputfile3 = open('/content/gdrive/MyDrive/outputfile2.txt', 'w')
lines = outputfile2.readlines()
for line in lines:
  outputfile3.write(line)

for file in os.listdir('/content/gdrive/MyDrive/dev-other'):
  filename = os.path.join('/content/gdrive/MyDrive/dev-other', file)
  for item1 in os.listdir(filename):
    filename = os.path.join(filename, item1)
    if os.path.isdir(filename):
      for item2 in os.listdir(filename):
        if(item2[-4:] == '.txt'):
          print(item2)

import collections
from speechbrain.utils.edit_distance import wer_summary, wer_details_for_batch, accumulatable_wer_stats

batches = [[[[1,2,3],[4,5,6]], [[1,2,4],[5,6]]]]
stats = collections.Counter()
for batch in batches:
    refs, hyps = batch
    stats = accumulatable_wer_stats(refs, hyps, stats)
    print(refs, '\n', hyps)
    print(stats)
print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))
