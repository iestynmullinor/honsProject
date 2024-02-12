from transformers import AutoTokenizer
import pandas as pd

#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Manually add [SEP] between sentences
# sentence1 = "This is the first sentence."

# sentence2 = "This is the second sentence."

# input_sequence = sentence1 + " [SEP] " + sentence2
# encoded_input = tokenizer(input_sequence, padding=True, truncation=True, max_length=512, return_tensors='pt')

# print(tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0]))


# we will convert climate fever training set to this format, then we will do the same for fever 

climate_fever_file_name = "climate_fever/labelled-training-data.csv"

with open(climate_fever_file_name, "r") as file:
    climate_fever_data = pd.read_csv(file)


# Drop the last column from the dataset
climate_fever_data = climate_fever_data.iloc[:, :-1]

claims= climate_fever_data["claim"]
evidence = climate_fever_data["evidence"]
formatted_labels = []

for index, row in climate_fever_data.iterrows():
    
    if row["label"] == "SUPPORTS" or row["label"] == "REFUTES":
        formatted_labels.append(0)
    else:
        formatted_labels.append(1)  



cf_reranker_training = pd.DataFrame({
    "claim": claims,
    "evidence": evidence,
    "label": formatted_labels
})

# save the formatted data
cf_reranker_training.to_csv("evidence_reranker/training_data/climate_fever_reranker_training.csv", index=False)

    