# HuggingFace demo code
import os
print( asr_model.transcribe_file('/speechbrain/testerfile.flac') )
outputfile = open('outputfile.txt', 'w')
print(* asr_model.transcribe_file(filename, InferenceDetailLevel.TOP_K_HYP) , sep='\n')
for filename in os.listdir('/content/data'):
  outputfile.write(str(*map(list, zip(* asr_model.transcribe_file('/content/data/'+filename, InferenceDetailLevel.TOP_K_HYP_SCORES) ))))
  print(outputfile)
# Equivalent w/ detail level enum
print(* asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav', InferenceDetailLevel.TOP1_HYP) )

# Return score for hypothesis also
print(* asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav', InferenceDetailLevel.TOP1_HYP_SCORES) )

# Get all details (token-level)
print(* asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav', InferenceDetailLevel.TOP1_HYP_DETAILS) )

# --- top-k results ---
# Top-k hypotheses (k=10, see yaml patch)
print(* asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav', InferenceDetailLevel.TOP_K_HYP) , sep='\n')
print(* asr_model.transcribe_file('/speechbrain/testerfile.wav', InferenceDetailLevel.TOP_K_HYP) , sep='\n')
# w/ scores
print(*map(list, zip(* asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav', InferenceDetailLevel.TOP_K_HYP_SCORES) )), sep='\n')

# w/ full details
print(*map(list, zip(* asr_model.transcribe_file('speechbrain/asr-crdnn-rnnlm-librispeech/example.wav', InferenceDetailLevel.TOP_K_HYP_DETAILS) )), sep='\n')

