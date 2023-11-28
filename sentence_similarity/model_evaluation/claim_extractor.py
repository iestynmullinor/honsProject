import json
import pickle
# adds all claims from climate fever to a pkl file

def extract_claims(file_path):
    claims = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'claim' in data:
                claims.append(data['claim'])
    return claims

# Use the function
claims = extract_claims('climate_fever/climate-fever-dataset-r1.jsonl')
#for claim in claims[:20]:
#    print(claim)

# Write to a pkl file
with open('climate_fever/claims.pkl', 'wb') as file:
    pickle.dump(claims, file)
