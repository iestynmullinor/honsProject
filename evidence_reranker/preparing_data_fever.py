import pandas as pd
from datasets import load_dataset

 # Load the dataset
dataset = load_dataset("pietrolesci/nli_fever")
# # Convert the DatasetDict to a DataFrame
df = pd.DataFrame(dataset['test'])

# # Save the DataFrame as a CSV file
df.to_csv('fever/nli_fever_test.csv', index=False)



# load fever/nlie_fever.csv as dataframe
fener_nli_df = pd.read_csv('fever/nli_fever_test.csv')

claims= fener_nli_df["premise"]
evidence = fener_nli_df["hypothesis"]
formatted_labels = []

for index, row in fener_nli_df.iterrows():
    
    if row["fever_gold_label"] == "SUPPORTS" or row["fever_gold_label"] == "REFUTES":
        formatted_labels.append(0)
    else:
        formatted_labels.append(1) 

fever_reranker_training = pd.DataFrame({
    "claim": claims,
    "evidence": evidence,
    "label": formatted_labels
})

# save the formatted data
fever_reranker_training.to_csv("evidence_reranker/training_data/fever_reranker_test.csv", index=False)


