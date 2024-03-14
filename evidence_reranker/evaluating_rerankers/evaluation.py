from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


rerankers = ["no-reranker",
              "roberta-reranker-fever-better",
              "roberta-reranker-climate-fever", 
              "roberta-reranker-f-cf",
              "roberta-reranker-f-cf-ipcc",
              "climatebert-rereranker-fever"]

# read in dictoinary evidence_reranker/evaluating_rerankers/nearest_for_each_claim.json
with open('evidence_reranker/evaluating_rerankers/nearest_for_each_claim.json', 'r', encoding='utf-8') as f:
    claim_with_evidence = json.load(f)
    claims = list(claim_with_evidence.keys())

# read in dictoinary evidence_reranker/evaluating_rerankers/valid_evidence.json
with open('evidence_reranker/evaluating_rerankers/valid_evidence.json', 'r', encoding='utf-8') as f:
    valid_evidence = json.load(f)

# read in dictoinary evidence_reranker/evaluating_rerankers/invalid_evidence.json
with open('evidence_reranker/evaluating_rerankers/invalid_evidence.json', 'r', encoding='utf-8') as f:
    invalid_evidence = json.load(f)


def rerank_evidence(model, tokenizer, claim_sentence, evidence_sentences):

    # if no model is provided, return the first 3 evidence sentences
    if model is None:
        return evidence_sentences[:3]

    sentence_rerank_scores = []

    for evidence_sentence in evidence_sentences:
        tokenized_input = tokenizer(claim_sentence, evidence_sentence, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
        model_output = model(**tokenized_input)

        # get the predicted class
        predicted_class = model_output.logits.argmax().item()
        #print(predicted_class)
        # get the probability
        #print(model_output.logits)
        probability = model_output.logits.max()

        # if predicted score is 0, add to sentence_rerank_scores
        if predicted_class == 0:
            sentence_rerank_scores.append((evidence_sentence, probability))

    # sort the sentence_rerank_scores by probability
    sentence_rerank_scores.sort(key=lambda x: x[1], reverse=True)

    # remove the probability from the sentence_rerank_scores
    sentence_rerank_scores = [x[0] for x in sentence_rerank_scores]

    # return the top 3 reranked evidence sentences, if there is less than 3 in the reranked list, return however many there is
    if len(sentence_rerank_scores) < 3:
        return sentence_rerank_scores
    else:
        return sentence_rerank_scores[:3]



if __name__ =="__main__":
    for i in range(len(rerankers)):
        print(f"{i}: {rerankers[i]}")

    #get model to evaluate
    model_index = int(input("Enter model index: "))
    model_name = rerankers[model_index]

    # uncomment this to reset evidence files
    # with open('evidence_reranker/evaluating_rerankers/valid_evidence.json', 'w', encoding='utf-8') as f:
    #     empty_valid_evidence = {claim: [] for claim in claims}
    #     json.dump(empty_valid_evidence, f, indent=4, ensure_ascii=False)

    # with open('evidence_reranker/evaluating_rerankers/invalid_evidence.json', 'w', encoding='utf-8') as f:
    #     empty_valid_evidence = {claim: [] for claim in claims}
    #     json.dump(empty_valid_evidence, f, indent=4, ensure_ascii=False)

    # load reranker model from huggingface
    # Load model directly
    if not model_name == "no-reranker":
        tokenizer = AutoTokenizer.from_pretrained(f"iestynmullinor/{model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(f"iestynmullinor/{model_name}")
    else:
        tokenizer = None
        model = None

    score_matrix = []
    count = 1
    for claim in claim_with_evidence.keys():
        evidence_sentences = rerank_evidence(model, tokenizer, claim, claim_with_evidence[claim])
        if len(evidence_sentences) == 0:
            count += 3
            claim_score = [0, 0, 0]
        else:
            claim_score = []
            for evidence in evidence_sentences:
                if evidence in valid_evidence[claim]:
                    claim_score.append(1)
                elif evidence in invalid_evidence[claim]:
                    claim_score.append(0)
                else:
                    firm = False
                    while not firm:
                        val = input(f"\n\n{count}/{len(claim_with_evidence.keys())*3}\nclaim: {claim}\nevidence: {evidence}\nIs this evidence valid? (y/n): ")
                        double_check = input(f"Are you sure? (y/n): ")
                        if double_check == 'y':
                            firm = True
                            if val == 'y':
                                claim_score.append(1)
                                valid_evidence[claim].append(evidence)
                            else:
                                claim_score.append(0)
                                invalid_evidence[claim].append(evidence)
                count += 1
            if len(claim_score) < 3:
                count += (3 - len(claim_score))
                claim_score.extend([0] * (3 - len(claim_score)))
        score_matrix.append(claim_score)

    #write the valid evidence json
    with open('evidence_reranker/evaluating_rerankers/valid_evidence.json', 'w', encoding='utf-8') as f:
        json.dump(valid_evidence, f, ensure_ascii=False)

    #write the invalid evidence json
    with open('evidence_reranker/evaluating_rerankers/invalid_evidence.json', 'w', encoding='utf-8') as f:
        json.dump(invalid_evidence, f, ensure_ascii=False)

    #write the score matrix
    with open(f'evidence_reranker/evaluating_rerankers/results/{model_name}_score_matrix.txt', 'w', encoding='utf-8') as f:
        json.dump(score_matrix, f, ensure_ascii=False)

