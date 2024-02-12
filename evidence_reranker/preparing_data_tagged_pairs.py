import json
import pandas as pd


# In this, we are creating a training set based on our annotations from evaluating different sentence transformer models.

valid_evidence_filename =  "sentence_similarity/model_evaluation/valid_evidence.json"

invalid_evidence_filename = "sentence_similarity/model_evaluation/invalid_evidence.json"

# load the evidence
with open(valid_evidence_filename, "r") as valid_evidence_file:
    valid_evidence = dict(json.load(valid_evidence_file))

with open(invalid_evidence_filename, "r") as invalid_evidence_file:
    invalid_evidence = dict(json.load(invalid_evidence_file))

claims = []

evidence = []

labels = []

for key in valid_evidence:
    for evidence_sentence in valid_evidence[key]:
        claims.append(key)
        evidence.append(evidence_sentence)
        labels.append(0)

for key in invalid_evidence:
    for evidence_sentence in invalid_evidence[key]:
        claims.append(key)
        evidence.append(evidence_sentence)
        labels.append(1)

# create a dataframe
training_data = pd.DataFrame({
    "claim": claims,
    "evidence": evidence,
    "label": labels
})

# shuffle the order of rows

training_data = training_data.sample(frac=1).reset_index(drop=True)

# save the dataframe as a csv file
training_data.to_csv("evidence_reranker/training_data/ipcc_cf_reranker_training.csv", index=False)


    

