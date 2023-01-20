!pip install sentencepiece transformers --quiet

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np

#changed from rubert, can try gpt?
model_name = 'bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#the fun stuff
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
  
#test example from librispeech
#the higher the score, the less confident
print(score(sentence='I inquired without knowing what his aunt\'s would be it was not true.', model=model, tokenizer=tokenizer)) 
print(score(sentence='I inquired without knowing what his answer would be it was not true.', model=model, tokenizer=tokenizer))
