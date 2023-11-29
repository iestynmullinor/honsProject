import requests
import pickle
import pandas as pd

# THIS AND CLAIM_EXTRACTOR.PY ARE THE SAME

data = pd.read_parquet('climate_fever/default_test_0000.parquet', engine='pyarrow')

claims = data['claim'].tolist()
print(len(claims))

with open ('climate_fever/claims.pkl', 'wb') as f:
    pickle.dump(claims, f)

# save as txt file as well
with open('climate_fever/claims.txt', 'w') as f:
    for claim in claims:
        f.write(claim + '\n')

