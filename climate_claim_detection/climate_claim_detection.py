from transformers import pipeline
from sklearn.metrics import classification_report

import time
import json


with open ('climate_fever/claims.json', 'r', encoding='utf-8') as f:
    CLIMATE_CLAIMS = json.load(f)

with open('fever/fever_claims.json', 'r', encoding='utf-8') as f:
    NON_CLIMATE_CLAIMS = json.load(f)

ALL_CLAIMS = CLIMATE_CLAIMS + NON_CLIMATE_CLAIMS

gold_labels = ['yes'] * len(CLIMATE_CLAIMS) + ['no'] * len(NON_CLIMATE_CLAIMS)

pipe_climatebert = pipeline("text-classification", model="climatebert/distilroberta-base-climate-detector")

start = time.time()
results_climatebert = pipe_climatebert(ALL_CLAIMS)
end = time.time()

print(f"Time taken for climatebert: {end - start}")

# since every claim is related to climate change, we check percentage of claims that are labeled as related

resulting_labels = [result['label'] for result in results_climatebert]

# Calculate precision, recall, and F1-score
report = classification_report(gold_labels, resulting_labels)

print(report)

# print all claims that are labeled as not related by mini
"""
print("Claims labeled as not related by mini:")
for i in range(len(results_mini)):
    if results_mini[i]['label'] == 'NOT_CLIMATE':
        print(CLAIMS[i])
        print()"""