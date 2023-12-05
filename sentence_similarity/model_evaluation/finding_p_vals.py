import numpy as np
from scipy.stats import chi2_contingency
import ast

file_path_of_matrix_1 = "sentence_similarity/model_evaluation/model_relevance_scores/all-mpnet-base-v2_score_matrix.txt"
file_path_of_matrix_2 = "sentence_similarity/model_evaluation/model_relevance_scores/all-roberta-large-v1_score_matrix.txt"


def chi_squared_test(model1_matrix, model2_matrix):
    # Create a contingency table
    contingency_table = np.array([
        [np.sum((model1_matrix == 1) & (model2_matrix == 1)), np.sum((model1_matrix == 1) & (model2_matrix == 0))],
        [np.sum((model1_matrix == 0) & (model2_matrix == 1)), np.sum((model1_matrix == 0) & (model2_matrix == 0))]
    ])

    print(contingency_table)

    # Perform chi-square test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p}")

    # Make a decision
    alpha = 0.05
    if p < alpha:
        print("Reject the null hypothesis. There is a significant difference.")
    else:
        print("Fail to reject the null hypothesis. No significant difference.")

if __name__ == "__main__":

    name_of_model_1 = (file_path_of_matrix_1.split('/')[-1].split('_')[0]).split('.')[0]
    name_of_model_2 = (file_path_of_matrix_2.split('/')[-1].split('_')[0]).split('.')[0]

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

    chi_squared_test(model1_matrix_claims_with_evidence, model2_matrix_claims_with_evidence)

    print("Chi-squared test number of relevant evidence:")

#    WE SORT THESE ROWS IN DESCENDING ORDER SO THAT ORDER OF EVIDENCE DOES NOT MATTER eg [1,1,0] becomes [1,1,0] and [1,0,1] becomes [1,1,0]
    model1_matrix_relevant_evidence = np.array([sorted(row, reverse=True) for row in matrix1]).flatten()
    model2_matrix_relevant_evidence = np.array([sorted(row, reverse=True) for row in matrix2]).flatten()

    chi_squared_test(model1_matrix_relevant_evidence, model2_matrix_relevant_evidence)