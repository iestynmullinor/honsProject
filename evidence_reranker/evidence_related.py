# Use a pipeline as a high-level helper
from transformers import pipeline
import time
import json

# this model could be better mwong/albert-base-climate-claim-related (more in the notion)

pipe = pipeline("text-classification", model="mwong/climatebert-base-f-climate-evidence-related")

claim = "The science is clear, climate change is making extreme weather events, including tornadoes, worse."

with open('sentence_similarity/data/sentence_section_pairs.json', 'r', encoding='utf-8') as f:
    SENTENCE_SECTION_PAIRS = json.load(f)
SENTENCES = [pair[0] for pair in SENTENCE_SECTION_PAIRS if len(pair[0]) < 500][:10]

#input_array = [f"[CLS] {claim} [SEP] {sentence} [SEP]" for sentence in SENTENCES]

input_array = [f"{claim} </s><s> {sentence}" for sentence in SENTENCES]


#input_array = ["The science is clear, climate change is making extreme weather events, including tornadoes, worse.</s></s>Climate change is making extreme weather events, including tornadoes, worse."]

start = time.time()
result = pipe(input_array)
end = time.time()
time_taken = end - start
print(f"Time taken: {time_taken}")

print(result)

valid_result_pairs = [(SENTENCES[i], result[i])  for i in range(len(result)) if result[i]['label'] == 'LABEL_1']
valid_result_pairs.sort(key=lambda x: x[1]['score'], reverse=True)

for pair in valid_result_pairs[:3]:
    print("sentence: ", pair[0])
    print("score: ", pair[1]['score'])
    print("label: ", pair[1]['label'])
    print()
