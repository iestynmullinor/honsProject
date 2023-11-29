import json
import pickle
import unicodedata
# adds all claims from climate fever to a pkl file

# THIS AND LOADING_DATASET.PY ARE THE SAME

def extract_claims(file_path):
    claims = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'claim' in data:
                claim = data['claim']
                claim = unicodedata.normalize('NFKD', claim).encode('ascii', 'ignore').decode('utf-8')
                claims.append(claim)
            
    return claims

# Use the function
claims = extract_claims('climate_fever/climate-fever-dataset-r1.jsonl')
#for claim in claims[:20]:
#    print(claim)

# Write to a pkl file
with open('climate_fever/claims.pkl', 'wb') as file:
    pickle.dump(claims, file)

# save as txt file as well
with open('climate_fever/claims.txt', 'w') as f:
    for claim in claims:
        f.write(claim + '\n')