# THIS CODE WILL BE MOVED TO THE GOOGLE COLAB NOTEBOOK
#  THERE HAS BEEN MANY CHANGES MADE TO THE COLAB ONE. PLEASE REFER TO THAT ONE


import json
from sklearn.metrics import classification_report
def get_concise_label(response):
      verdict = response["choices"][0]["text"]

      if "refuted" in verdict.split("\n")[-1]:
        result = "REFUTES"

      elif "supported" in verdict.split("\n")[-1]:
        result = "SUPPORTS"

      elif "not_enough_info" in verdict.split("\n")[-1]:
        result = "NOT_ENOUGH_INFO"

      else:
        result = "ERROR"

      return result


def get_prompt(claim, evidence_text):
    prompt = f'''
        Claim: {claim}
        Evidence: {evidence_text}
        Class:
        '''


    prompt_template=f'''SYSTEM: Classify the claim/evidence pair into one of the classes. Respond with the name of the class which best describes if the evidence supports the claim (supports), refutes the claim (refutes), or if there is not enough information (not_enough_info). Do not provide an explanation. Your entire response is limited to the following three options: [`supported`,`refuted`,`not_enough_info`].

                        Classes: [`supports`, `refutes`, `not_enough_info`]

                        Claim: Ice berg melts, ocean level remains the same.
                        Evidence: The melting of floating ice raises the ocean level.
                        Class: `refutes`

                        Claim: Global warming is driving polar bears toward extinction.
                        Evidence: Environmental impacts include the extinction or relocation of many species as their ecosystems change, most immediately the environments of coral reefs, mountains, and the Arctic.
                        Class: `supports`

                        Claim: There is no statistical evidence that global warming is intensifying floods.
                        Evidence: Such events will continue to occur more often and with greater intensity.
                        Class: `not_enough_info`

                        USER: {prompt}

                        ASSISTANT:
                        '''
    return prompt_template

def get_macro_label(micro_labels):
    if "REFUTES" in micro_labels and "SUPPORTS" in micro_labels:
        return "DISPUTED"
    elif "REFUTES" in micro_labels:
        return "REFUTES"
    elif "SUPPORTS" in micro_labels:
        return "SUPPORTS"
    else:
        return "NOT_ENOUGH_INFO"


with open('climate_fever/macro_labels.json', 'r', encoding='utf-8') as f:
    gold_macro_labels = json.load(f)

with open('climate_fever/micro_labels.json', 'r', encoding='utf-8') as f:
    gold_micro_labels = json.load(f)
    gold_micro_labels = [item for sublist in gold_micro_labels for item in sublist]
    

with open('climate_fever/climate-fever-dataset-r1.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f.readlines()]

with open('climate_fever/micro_label_predictions.json', 'r', encoding='utf-8') as f:
    number_of_predictions_so_far = len(json.load(f))
    micro_label_predictions = json.load(f)

macro_label_predictions = []
claim_evidence_pairs = []


# generating claim evidence pairs
for i,line in enumerate(data):
    print(f"Processing claim {i+1} of {len(data)}")
    claim = line["claim"]
    for evidence in line["evidences"]:
        evidence_text = evidence["evidence"]
        claim_evidence_pairs.append({"claim": claim, "evidence": evidence_text})
        

for claim_evidence_pair in claim_evidence_pairs[number_of_predictions_so_far:]:
    
    print(f"Processing pair {len(micro_label_predictions)+1} of {len(claim_evidence_pairs)}")

    claim = claim_evidence_pair["claim"]
    evidence_text = claim_evidence_pair["evidence"]
    prompt = get_prompt(claim, evidence_text)

    #response=lcpp_llm(prompt=prompt, max_tokens=150, temperature=0, top_p=0.95,
    #          repeat_penalty=1.2, top_k=150,
    #          echo=True)

    response = "response"
    # this is replaces with above commmented code in the google colab notebook
        
    result = get_concise_label(response)
    micro_label_predictions.append(result)
    with open('climate_fever/micro_label_predictions.json', 'w') as f:
        json.dump(result, f)
        f.write("\n")
    

# generating macro label predictions
for i in range(0, len(micro_label_predictions), 5):
    micro_labels_for_claim = micro_label_predictions[i:i+5]
    macro_label_prediction = get_macro_label(micro_labels_for_claim)
    macro_label_predictions.append(macro_label_prediction)



micro_label_results = classification_report(gold_micro_labels, micro_label_predictions)
macro_label_results = classification_report(gold_macro_labels, macro_label_predictions)

print("results for micro labels:")
print(micro_label_results)

print("results for macro labels:")
print(macro_label_results)

with open('climate_fever/baseline_results.txt', 'w') as f:
    f.write("Results for micro labels:\n")
    f.write(micro_label_results)
    f.write("\n\nResults for macro labels:\n")
    f.write(macro_label_results)




