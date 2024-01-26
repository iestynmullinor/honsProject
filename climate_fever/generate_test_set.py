import random
import json

with open('climate_fever/climate-fever-dataset-r1.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f.readlines()]


test_set_size = 95
random.seed(42)
random.shuffle(data)
test_set = data[:test_set_size]
train_set = data[test_set_size:]

with open('climate_fever/climate-fever-train.jsonl', 'w', encoding='utf-8') as f:
    for line in train_set:
        json.dump(line, f, ensure_ascii=False)
        f.write("\n")

with open('climate_fever/climate-fever-test.jsonl', 'w', encoding='utf-8') as f:
    for line in test_set:
        json.dump(line, f, ensure_ascii=False)
        f.write("\n")
