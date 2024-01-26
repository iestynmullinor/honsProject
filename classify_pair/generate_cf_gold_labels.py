import json

with open('climate_fever/climate-fever-train.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f.readlines()]

macro_labels = [line["claim_label"] for line in data]

with open('climate_fever/train_macro_labels.json', 'w', encoding='utf-8') as f:
    json.dump(macro_labels, f, ensure_ascii=False, indent=4)

micro_labels = []
for line in data:
    evidence_labels = []
    for evidence in line["evidences"]:
        evidence_labels.append(evidence["evidence_label"])
    micro_labels.append(evidence_labels)

with open('climate_fever/train_micro_labels.json', 'w', encoding='utf-8') as f:
    json.dump(micro_labels, f, ensure_ascii=False, indent=4)

