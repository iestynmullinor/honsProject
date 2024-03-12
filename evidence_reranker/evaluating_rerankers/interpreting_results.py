import ast


def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        list_of_lists = ast.literal_eval(data)
    return list_of_lists

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

# open evidence_reranker/evaluating_rerankers/results/no-reranker_score_matrix.txt
score_matrix = read_file('evidence_reranker/evaluating_rerankers/results/roberta-reranker-fever-better_score_matrix.txt')

print("backed up claims:", get_percentage_of_claims_with_evidence(score_matrix))

print("relevant evidence:", get_percentage_of_relevant_evidence(score_matrix))