from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import json
from transformers import pipeline

DETECT_CLIMATE_CLAIM = True
RERANK = True

# this mode is actually very good
CLIMATE_CLAIM_DETECTOR_MODEL = "climatebert/distilroberta-base-climate-detector"

# this is confirmed to be the best one from testing
SENTENCE_TRANSFORMERS_MODEL = 'all-mpnet-base-v2'
K = 30
RADIUS = 1

# this has not been confirmed to be any good
RERANKER_MODEL ="mwong/climatebert-base-f-climate-evidence-related"

with open('sentence_similarity/data/sentence_section_pairs.json', 'r', encoding='utf-8') as f:
    SENTENCE_SECTION_PAIRS = json.load(f)
SENTENCES = [pair[0] for pair in SENTENCE_SECTION_PAIRS]
SECTION_TITLES = [pair[1] for pair in SENTENCE_SECTION_PAIRS]

with open(f'sentence_similarity/model_evaluation/model_embeddings/MODEL_{SENTENCE_TRANSFORMERS_MODEL}_EMBEDDINGS.pkl', 'rb') as f:
    EMBEDDINGS = pickle.load(f)

reports = {'synthesis' : 'Synthesis Report', 
           'wg1' : 'Working Group 1 - Climate Change 2021: The Physical Science Basis',
           'wg2' : 'Working Group 2 - Climate Change 2022: Impacts, Adaptation and Vulnerability',
           'wg3' : 'Working Group 3 - Climate Change 2022: Mitigation of Climate Change'}

nn = NearestNeighbors(n_neighbors=K, metric='cosine', radius=RADIUS)
nn.fit(EMBEDDINGS)

sentence_similarity_model = SentenceTransformer(SENTENCE_TRANSFORMERS_MODEL)
reranker = pipeline("text-classification", model=RERANKER_MODEL)
claim_detector = pipeline("text-classification", model=CLIMATE_CLAIM_DETECTOR_MODEL)


def rerank(evidence_sentences):
    model_output = reranker(evidence_sentences)
    valid_result_pairs = [(evidence_sentences[i], model_output[i])  for i in range(len(model_output)) if model_output[i]['label'] == 'LABEL_1']
    valid_result_pairs.sort(key=lambda x: x[1]['score'], reverse=True)
    return valid_result_pairs

def format_section(section):
    split_section = section.split('/')[1:]
    document_encoded = split_section[0]
    title = split_section[-1]
    document = reports[document_encoded]
    formatted_section = f"REPORT: {document}\nSECTION TITLE: {title}\n"
    return formatted_section


if __name__=="__main__":
    while True:
        input_sentence = input("Enter sentence: ")

        if DETECT_CLIMATE_CLAIM:
            claim_detector_output = claim_detector([input_sentence])
            if claim_detector_output[0]['label'] == 'no':
                print("This sentence is not a climate claim, so no evidence will be retrieved.")
                continue
        

        input_embedding = sentence_similarity_model.encode(input_sentence)

        distances, indices = nn.kneighbors([input_embedding])

        evidence_sentences = [SENTENCES[index] for index in indices[0]]
        # print the 5 nearest neighbors
        print("5 Nearest Neighbors Before Reranking:")
        for distance,index in zip(distances[0],indices[0][:5]):
            print("EVIDENCE:", SENTENCES[index])
            print(format_section(SECTION_TITLES[index]))
            print("DISTANCE:", distance)
            print("------------------------------------------------------------------------------------------------------------------------")

        
        if RERANK:
            reranked_evidence_sentences = rerank(evidence_sentences)
            print("5 Nearest Neighbors After Reranking:")
            for pair in reranked_evidence_sentences[:5]:
                print("EVIDENCE:", pair[0])
                print(format_section(SECTION_TITLES[SENTENCES.index(pair[0])]))
                print("SCORE:", pair[1]['score'])
                print("------------------------------------------------------------------------------------------------------------------------")

