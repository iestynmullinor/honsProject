import pandas as pd 
import json

instruction = """Categorize the claim/evidence pair into one of the 3 categories based on if the evidence supports or refutes the claim, or if there is not enough information to make a decision:


SUPPORTS
REFUTES
NOT_ENOUGH_INFO

"""

with open('climate_fever/climate-fever-test.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f.readlines()]

claim_evidence_pairs = []

for line in data:
    claim = line["claim"]
    evidences = [evidence["evidence"] for evidence in line["evidences"]]
    labels = [evidence["evidence_label"] for evidence in line["evidences"]]

    for evidence, label in zip(evidences, labels):
        claim_evidence_pairs.append({"claim": claim, "evidence": evidence, "label": label })

df = pd.DataFrame(claim_evidence_pairs)
df['instruction'] = instruction

# Converting to list of dictionaries
train_json = df.to_json(orient = 'records', lines = True).splitlines()

# Saving as a JSON file
with open(f"climate_fever/labelled-test-data.jsonl", 'w') as f:
    for line in train_json:
        f.write(f"{line}\n")

# Saving as a CSV file
df.to_csv(f"climate_fever/labelled-test-data.csv", index = False)