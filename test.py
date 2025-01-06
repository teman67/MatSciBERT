import torch
from normalize_text import normalize
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
model = AutoModel.from_pretrained('m3rg-iitd/matscibert')

sentences = ['SiO2 is a network former.']
norm_sents = [normalize(s) for s in sentences]
tokenized_sents = tokenizer(norm_sents)
tokenized_sents = {k: torch.Tensor(v).long() for k, v in tokenized_sents.items()}

with torch.no_grad():
    last_hidden_state = model(**tokenized_sents)[0]