import json

claims = []
with open('fever/fever.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        claim = data['claim']
        if "climate" not in claim.lower():
            claims.append(claim)
        if len(claims) == 1535:
            break

with open('fever/fever_claims.json', 'w', encoding='utf-8') as file:
    json.dump(claims, file, ensure_ascii=False)


