import json

NO_CLAIMS = 50

MODEL_NAMES_WITHOUT_DIR = ['all-mpnet-base-v2', 
          'distilroberta-base-climate-f', 
          'all-MiniLM-L12-v2',
          'all-roberta-large-v1',
          'bge-base-en-v1.5',
          'all-MiniLM-L6-v2',
          'climate-fever-msmarco-distilbert-gpl', #sentence transformers version of climate-bert
          'bpr-gpl-climate-fever-base-msmarco-distilbert-tas-b',
          'e5-large-v2',
          'gte-large']



if __name__=="__main__":

    for i in range(len(MODEL_NAMES_WITHOUT_DIR)):
        print(f"{i}: {MODEL_NAMES_WITHOUT_DIR[i]}")

    #get model to evaluate
    model_index = int(input("Enter model index: "))
    model_name = MODEL_NAMES_WITHOUT_DIR[model_index]

    # load claims with evidence
    with open(f'sentence_similarity/model_evaluation/model_relevance_evaluation/{model_name}_k_nearest_for_all_claims.txt', 'r', encoding='utf-8') as f:
        claim_with_evidence = json.load(f)

    # UNCOMMENT THIS TO RESET EVIDENCE FILES

    #with open('sentence_similarity/model_evaluation/valid_evidence.json', 'w', encoding='utf-8') as f:
    #    empty_valid_evidence = {claim: [] for claim in claim_with_evidence.keys()}
    #    json.dump(empty_valid_evidence, f, indent=4, ensure_ascii=False)

    #with open('sentence_similarity/model_evaluation/invalid_evidence.json', 'w', encoding='utf-8') as f:
    #    empty_valid_evidence = {claim: [] for claim in claim_with_evidence.keys()}
    #    json.dump(empty_valid_evidence, f, indent=4, ensure_ascii=False)

    #get the valid evidence json
    with open('sentence_similarity/model_evaluation/valid_evidence.json', 'r', encoding='utf-8') as f:
        valid_evidence = json.load(f)

    #get the invalid evidence json
    with open('sentence_similarity/model_evaluation/invalid_evidence.json', 'r', encoding='utf-8') as f:
        invalid_evidence = json.load(f)

    score_matrix = []
    count = 1
    for claim in claim_with_evidence.keys():
        claim_score = []
        for evidence in claim_with_evidence[claim]:
            if evidence in valid_evidence[claim]:
                claim_score.append(1)
            elif evidence in invalid_evidence[claim]:
                claim_score.append(0)
            else:
                firm = False
                while not firm:
                    val = input(f"\n\n{count}/{NO_CLAIMS*3}\nclaim: {claim}\nevidence: {evidence}\nIs this evidence valid? (y/n): ")
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
        score_matrix.append(claim_score)

    #write the valid evidence json
    with open('sentence_similarity/model_evaluation/valid_evidence.json', 'w', encoding='utf-8') as f:
        json.dump(valid_evidence, f, ensure_ascii=False)

    #write the invalid evidence json
    with open('sentence_similarity/model_evaluation/invalid_evidence.json', 'w', encoding='utf-8') as f:
        json.dump(invalid_evidence, f, ensure_ascii=False)

    #write the score matrix
    with open(f'sentence_similarity/model_evaluation/model_relevance_scores/{model_name}_score_matrix.txt', 'w', encoding='utf-8') as f:
        json.dump(score_matrix, f, ensure_ascii=False)

    
    

    
