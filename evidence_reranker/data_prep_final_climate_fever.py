import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# # Load the dataset
# dataset = load_dataset("pietrolesci/nli_fever")
# # # Convert the DatasetDict to a DataFrame
# df = pd.DataFrame(dataset['train'])

# # # Save the DataFrame as a CSV file
# df.to_csv('fever/nli_fever_test.csv', index=False)

# load climate_fever/labelled-test-data.csv as dataframe

with open('climate_fever/labelled-test-data.csv', 'r') as f:
    climate_fever_test_df = pd.read_csv(f)
    # drop the 'instruction' column
    climate_fever_test_df = climate_fever_test_df.drop(columns=["instruction"])

# load climate_fever/labelled-training-data.csv as dataframe
with open('climate_fever/labelled-training-data.csv', 'r') as f:
    climate_fever_train_df = pd.read_csv(f)
    # drop the 'instruction' column
    climate_fever_train_df = climate_fever_train_df.drop(columns=["instruction"])


def main(TEST_SET):

    REMOVE_NEI = True

    if TEST_SET:
        # load fever/nlie_fever.csv as dataframe
        fener_nli_df = climate_fever_test_df
    # load fever/nlie_fever.csv as dataframe
        
    else:
        fener_nli_df = climate_fever_train_df

    if REMOVE_NEI:
        fener_nli_df = fener_nli_df[fener_nli_df["label"] != "NOT_ENOUGH_INFO"]

    claims= fener_nli_df["claim"]
    evidence = fener_nli_df["evidence"]
    formatted_labels = []

    for index, row in fener_nli_df.iterrows():
        
        formatted_labels.append(0) 


    fever_reranker_training = pd.DataFrame({
        "claim": claims,
        "evidence": evidence,
        "label": formatted_labels
    })

    # remove all rows where evidence is empty 
    fever_reranker_training = fever_reranker_training[fever_reranker_training["evidence"] != ""]



    # find the number of irrelevant evidence
    irrelevant_evidence = fever_reranker_training[fever_reranker_training["label"] == 1]

    # find the number of relevant evidence
    relevant_evidence = fever_reranker_training[fever_reranker_training["label"] == 0]

    # find the difference between the number of relevant and irrelevant evidence


    difference = len(relevant_evidence) - len(irrelevant_evidence)
    print(difference)
    # now we balance the dataset by adding more irrelevant claim/evidence pairs
    # we do this by adding existing claims, and evidence for another claim to the dataset, making sure that the evidence is irrelevant

    new_rows = []
    for _ in tqdm(range(difference)):
        
        # pick a random row from the dataframe for the claim
        random_row_for_claim = fever_reranker_training.sample()


        # get the claim from the random row
        claim = random_row_for_claim["claim"].values[0]

        # pick a random row from the dataframe for the evidence
        random_row_for_evidence = fever_reranker_training.sample()


        # make sure that the claim for this row is not the same as "claim"

        while random_row_for_evidence["claim"].values[0] == claim:
            random_row_for_evidence = fever_reranker_training.sample()

        

        # get the evidence from the random row
        evidence = random_row_for_evidence["evidence"].values[0]

        # add the claim, evidence and label to the new_rows list
        new_rows.append([claim, evidence, 1])



    # create a dataframe from the new_rows list
    new_rows_df = pd.DataFrame(new_rows, columns=["claim", "evidence", "label"])

    # add the new rows to the fever_reranker_training dataframe
    fever_reranker_training = pd.concat([fever_reranker_training, new_rows_df], ignore_index=True)

    # print the number of rows with label 0 and 1
    print(fever_reranker_training["label"].value_counts())




    # shuffle the order of rows
    fever_reranker_training = fever_reranker_training.sample(frac=1).reset_index(drop=True)
    
    print(len(fever_reranker_training))
    # save the formatted data

    if TEST_SET:
        
        fever_reranker_test = fever_reranker_training

        fever_reranker_test.to_csv("evidence_reranker/training_data/climate_fever_reranker_test.csv", index=False)

    else:
        fever_reranker_training.to_csv("evidence_reranker/training_data/climate_fever_reranker_train.csv", index=False)

if __name__ == "__main__":
    main(False)
    main(True)