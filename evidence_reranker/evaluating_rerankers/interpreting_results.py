import ast
from tabulate import tabulate

model_names = ["no-reranker",
               "roberta-reranker-f-cf",
               "roberta-reranker-fever-better",
               "roberta-reranker-f-cf-ipcc",
               "climatebert-rereranker-fever",
               "climatebert-rereranker-f-cf",
               "climatebert-rereranker-f-cf-ipcc"]

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

model_results = {}
for model_name in model_names:
    file_name = f'evidence_reranker/evaluating_rerankers/results/{model_name}_score_matrix.txt'
    score_matrix = read_file(file_name)
    model_results[model_name] = score_matrix

backed_up_claims = {}
relevant_evidence = {}

for model_name in model_results.keys():
    backed_up_claims[model_name] = get_percentage_of_claims_with_evidence(model_results[model_name])
    relevant_evidence[model_name] = get_percentage_of_relevant_evidence(model_results[model_name])

# Print as a readable table
table_data = []
for model_name in model_names:
    table_data.append([model_name, backed_up_claims[model_name], relevant_evidence[model_name]])

print(tabulate(table_data, headers=["Model Name", "Backed Up Claims", "Relevant Evidence"], tablefmt="fancy_grid"))

# save the table to evidence_reranker/evaluating_rerankers/results_table.txt

with open("evidence_reranker/evaluating_rerankers/results_table.txt", "w") as file:
    file.write(tabulate(table_data, headers=["Model Name", "Backed Up Claims", "Relevant Evidence"], tablefmt="fancy_grid"))
