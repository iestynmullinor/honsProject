import json
import ast
from tabulate import tabulate

MODEL_NAMES = ['all-mpnet-base-v2', 
          #'climatebert/distilroberta-base-climate-f', # THIS IS NOT A SENTEMNCE TRANSFORMERS MODEL LEAVE IT OUT
          'all-MiniLM-L12-v2',
          'all-roberta-large-v1',
          'bge-base-en-v1.5',
          'all-MiniLM-L6-v2',
          'climate-fever-msmarco-distilbert-gpl', #sentence transformers version of climate-bert
          'bpr-gpl-climate-fever-base-msmarco-distilbert-tas-b',
          'e5-large-v2',
          'gte-large',
          'bm25',
          'contriever',
          'contriever-msmarco'
          ]


def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        list_of_lists = ast.literal_eval(data)
    return list_of_lists

def get_score_matrices():
    eval_dict = {}
    for model_name in MODEL_NAMES:
        file_name = f'sentence_similarity/model_evaluation/model_relevance_scores/{model_name}_score_matrix.txt'
        score_matrix = read_file(file_name)
        eval_dict[model_name] = score_matrix
    return eval_dict

def get_percentage_of_claims_with_evidence(matrix):
    no_claims_with_evidence = 0
    for row in matrix:
        if 1 in row:
            no_claims_with_evidence += 1
    return no_claims_with_evidence / len(matrix)

def get_percentage_of_relevant_evidence(matrix):
    no_relevant_evidence = 0
    flatted_matrix = [item for sublist in matrix for item in sublist]
    for item in flatted_matrix:
        if item == 1:
            no_relevant_evidence += 1
    return no_relevant_evidence / len(flatted_matrix)


if __name__ == "__main__":
    model_results = get_score_matrices()
    backed_up_claims = {}
    relevant_evidence = {}
    for model_name in model_results.keys():
        backed_up_claims[model_name] = get_percentage_of_claims_with_evidence(model_results[model_name])
        relevant_evidence[model_name] = get_percentage_of_relevant_evidence(model_results[model_name])

    # Print as a readable table
    table_data = []
    for model_name in MODEL_NAMES:
        table_data.append([model_name, backed_up_claims[model_name], relevant_evidence[model_name]])

    headers = ["Model Name", "Backed Up Claims", "Relevant Evidence"]
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

    # Write table to a txt file
    with open("sentence_similarity/model_evaluation/model_evaluation_table.txt", "w") as file:
        file.write(tabulate(table_data, headers=headers))
        
    
    