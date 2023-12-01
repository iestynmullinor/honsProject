import requests
import pickle
import pandas as pd
import json

# DON'T USE THIS ONE

data = pd.read_parquet('climate_fever/default_test_0000.parquet', engine='pyarrow')

claims = data['claim'].tolist()
print(len(claims))

with open('climate_fever/claims.pkl', 'wb') as f:
    pickle.dump(claims, f)

# Save as txt file
with open('climate_fever/claims.txt', 'w') as f:
    for claim in claims:
        f.write(claim + '\n')

# Save as json file
with open('climate_fever/claims.json', 'w') as f:
    json.dump(claims, f)


