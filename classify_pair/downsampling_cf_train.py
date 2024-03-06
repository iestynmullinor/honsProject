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

num_supports = label_counts['SUPPORTS']
num_refutes = label_counts['REFUTES']
num_nei = label_counts['NOT_ENOUGH_INFO']

# Downsample the "NOT_ENOUGH_INFO" label
nei_df = df[df['label'] == 'NOT_ENOUGH_INFO']
nei_downsampled = resample(nei_df, replace=False, n_samples=num_refutes, random_state=42)

# Downsample the "SUPPORTS" label
supports_df = df[df['label'] == 'SUPPORTS']
supports_downsampled = resample(supports_df, replace=False, n_samples=num_refutes, random_state=42)

# Concatenate the downsampled dataframes
df_downsampled = pd.concat([nei_downsampled, supports_downsampled, df[df['label'] == 'REFUTES']])
print(df_downsampled['label'].value_counts())

# shuffle the dataframe
df_downsampled = df_downsampled.sample(frac=1).reset_index(drop=True)

# Save the downsampled dataframe as a CSV file
df_downsampled.to_csv('climate_fever/downsampled-labelled-training-cf.csv', index=False)

