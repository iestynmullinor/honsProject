import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import ast
from tabulate import tabulate

BEST_MODEL = "sentence_similarity/model_evaluation/model_relevance_scores/all-mpnet-base-v2_score_matrix.txt"
OTHER_MODELS = ["all-roberta-large-v1_score_matrix.txt",
                "all-MiniLM-L6-v2_score_matrix.txt",
                "all-MiniLM-L12-v2_score_matrix.txt",
                "bge-base-en-v1.5_score_matrix.txt",
                "bpr-gpl-climate-fever-base-msmarco-distilbert-tas-b_score_matrix.txt",
                "climate-fever-msmarco-distilbert-gpl_score_matrix.txt",
                "e5-large-v2_score_matrix.txt",
                "gte-large_score_matrix.txt",
                "bm25_score_matrix.txt",
                'contriever_score_matrix.txt',
                'contriever-msmarco_score_matrix.txt'



]


                
                

def mc_nemer_test(model1_matrix, model2_matrix):
    # Create a contingency table
    contingency_table = np.array([
        [np.sum((model1_matrix == 1) & (model2_matrix == 1)), np.sum((model1_matrix == 1) & (model2_matrix == 0))],
        [np.sum((model1_matrix == 0) & (model2_matrix == 1)), np.sum((model1_matrix == 0) & (model2_matrix == 0))]
    ])

    print(contingency_table)

    # Perform McNemer's test
    result = mcnemar(contingency_table, exact=True)

    print(result)

    return result

def perform_test(file_path_of_matrix_1, file_path_of_matrix_2):

    name_of_model_1 = file_path_of_matrix_1.split('/')[-1].split('.txt')[0].split('_')[0]
    name_of_model_2 = file_path_of_matrix_2.split('/')[-1].split('.txt')[0].split('_')[0]

    print(f"Comparing {name_of_model_1} and {name_of_model_2}:")

    with open(file_path_of_matrix_1, 'r') as file:
        data = file.read()
        matrix1 = ast.literal_eval(data)

    with open(file_path_of_matrix_2, 'r') as file:
        data = file.read()
        matrix2 = ast.literal_eval(data)

    print("Chi-squared test number of claims with evidence:")

    
    model1_matrix_claims_with_evidence = np.array([1 if 1 in row else 0 for row in matrix1])
    model2_matrix_claims_with_evidence = np.array([1 if 1 in row else 0 for row in matrix2])

    claims_with_evidence_result=mc_nemer_test(model1_matrix_claims_with_evidence, model2_matrix_claims_with_evidence)

    print("Chi-squared test number of relevant evidence:")

#    WE SORT THESE ROWS IN DESCENDING ORDER SO THAT ORDER OF EVIDENCE DOES NOT MATTER eg [1,1,0] becomes [1,1,0] and [1,0,1] becomes [1,1,0]
    model1_matrix_relevant_evidence = np.array([sorted(row, reverse=True) for row in matrix1]).flatten()
    model2_matrix_relevant_evidence = np.array([sorted(row, reverse=True) for row in matrix2]).flatten()

    valid_evidence_result=mc_nemer_test(model1_matrix_relevant_evidence, model2_matrix_relevant_evidence)

    return claims_with_evidence_result, valid_evidence_result

if __name__=="__main__":
    results_dict = {'models': [], 'P-Value comaparing Claims With Evidence': [], 'P-Value comparing amount of Valid Evidence': []}
    for model in OTHER_MODELS:
        claims_with_evidence_result, valid_evidence_result = perform_test(BEST_MODEL, f"sentence_similarity/model_evaluation/model_relevance_scores/{model}")
        results_dict['models'].append(model.split('.txt')[0].split('_')[0])
        results_dict['P-Value comaparing Claims With Evidence'].append(claims_with_evidence_result.pvalue)
        results_dict['P-Value comparing amount of Valid Evidence'].append(valid_evidence_result.pvalue)

    table = tabulate(results_dict, headers='keys', tablefmt='fancy_grid')
    print("All Models Compared to Best Model (all-mpnet-base-v2):")
    print(table)

    with open("sentence_similarity/model_evaluation/all_p_vals.txt", "w") as file:
        file.write(table)
