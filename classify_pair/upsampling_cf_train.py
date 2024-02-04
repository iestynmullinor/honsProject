import json
import pandas as pd
from sklearn.utils import resample

with open('climate_fever/labelled-training-data.csv', 'r', encoding='utf-8') as f:
    df = pd.read_csv('climate_fever/labelled-training-data.csv', encoding='utf-8')

label_counts = df['label'].value_counts()
print(label_counts)

#NOT_ENOUGH_INFO    4625
#SUPPORTS           1818
#REFUTES             757

# Upsample the "SUPPORTS" label
supports_df = df[df['label'] == 'SUPPORTS']
supports_upsampled = resample(supports_df, replace=True, n_samples=label_counts['NOT_ENOUGH_INFO']-label_counts['SUPPORTS'], random_state=42)

# Upsample the "REFUTES" label
refutes_df = df[df['label'] == 'REFUTES']
refutes_upsampled = resample(refutes_df, replace=True, n_samples=label_counts['NOT_ENOUGH_INFO']-label_counts["REFUTES"], random_state=42)

# Concatenate the upsampled dataframes
df_upsampled = pd.concat([df, supports_upsampled, refutes_upsampled])

# Check the label counts after upsampling
label_counts_upsampled = df_upsampled['label'].value_counts()
print(label_counts_upsampled)

# Save the upsampled dataframe as a JSONL file
df_upsampled.to_json('upsampled-labelled-training-cf.jsonl', orient='records', lines=True)

# Save the upsampled dataframe as a CSV file
df_upsampled.to_csv('upsampled-labelled-training-cf.csv', index=False)

