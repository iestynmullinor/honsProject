import pandas as pd 

file_name = "fever/nli_fever_dev.csv"

df = pd.read_csv(file_name)

df = df[["premise", "hypothesis", "fever_gold_label"]]

df = df.rename(columns={"premise": "claim", "hypothesis": "evidence", "fever_gold_label": "label"})

df["label"] = df["label"].replace("NOT ENOUGH INFO", "NOT_ENOUGH_INFO")

# Create a new dataframe with 2000 samples of each label
supports_df = df[df["label"] == "SUPPORTS"].sample(n=50, replace=False)
refutes_df = df[df["label"] == "REFUTES"].sample(n=50, replace=False)
not_enough_info_df = df[df["label"] == "NOT_ENOUGH_INFO"].sample(n=50, replace=False)

# Concatenate the dataframes
new_df = pd.concat([supports_df, refutes_df, not_enough_info_df])

# Reset the index of the new dataframe
new_df = new_df.reset_index(drop=True)

# shuffle the dataframe
new_df = new_df.sample(frac=1).reset_index(drop=True)

# Save the new dataframe to a csv file
new_df.to_csv("classify_pair/data/fever_test_for_llm.csv", index=False)