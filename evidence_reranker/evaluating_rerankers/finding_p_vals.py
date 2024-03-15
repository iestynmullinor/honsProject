import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import ast
from tabulate import tabulate

BEST_MODEL = "evidence_reranker/evaluating_rerankers/results/roberta-reranker-fever-better_score_matrix.txt"
OTHER_MODELS = ["roberta-reranker-f-cf_score_matrix.txt",
                "roberta-reranker-f-cf-ipcc_score_matrix.txt",
                "climatebert-rereranker-fever_score_matrix.txt",
                "climatebert-rereranker-f-cf_score_matrix.txt",
                "climatebert-rereranker-f-cf-ipcc_score_matrix.txt",
                "no-reranker_score_matrix.txt"


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
        claims_with_evidence_result, valid_evidence_result = perform_test(BEST_MODEL, f"evidence_reranker/evaluating_rerankers/results/{model}")
        results_dict['models'].append(model.split('.txt')[0].split('_')[0])
        results_dict['P-Value comaparing Claims With Evidence'].append(claims_with_evidence_result.pvalue)
        results_dict['P-Value comparing amount of Valid Evidence'].append(valid_evidence_result.pvalue)

    table = tabulate(results_dict, headers='keys', tablefmt='fancy_grid')
    print("All Models Compared to Best Model (RoBERTa-FEVER):")
    print(table)

    with open("evidence_reranker/evaluating_rerankers/all_p_vals.txt", "w") as file:
        file.write(table)
