from transformers import AutoTokenizer, AutoModel
import torch
import json
import pickle
import numpy as np

CLIMATE_FEVER = True
MODEL = "facebook/contriever-msmarco"

if CLIMATE_FEVER:
    extra_bit1="_cf"
    extra_bit2="_CF"
else:
    extra_bit1=""
    extra_bit2=""

with open("sentence_similarity/data/sentence_section_pairs.json", "r", encoding="utf-8") as f:
    SENTENCE_SECTION_PAIRS = json.load(f)
sentences = [pair[0] for pair in SENTENCE_SECTION_PAIRS]

with open("climate_fever/claims.json", "r", encoding="utf-8") as f:
    CLIMATE_FEVER_CLAIMS = json.load(f)

if CLIMATE_FEVER:
    sentences = CLIMATE_FEVER_CLAIMS

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings



tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

embeddings = []

i=1
for sentence in sentences:

    print(f"{i}/{len(sentences)}")
    i+=1

    # Apply tokenizer
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
    outputs = model(**inputs)

    embedding_as_tensor = mean_pooling(outputs[0], inputs['attention_mask'])


    embedding = embedding_as_tensor.tolist()[0]
    embeddings.append(embedding)

print(np.array(embeddings).shape)

with open(f"contriever_&_climatebert/contriever{extra_bit1}_embeddings/MODEL_{MODEL.split('/')[1]}{extra_bit2}_EMBEDDINGS.pkl", "wb") as f:
    pickle.dump(embeddings, f)

