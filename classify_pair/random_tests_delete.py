import json

def get_macro_label(micro_labels):
    if "REFUTES" in micro_labels and "SUPPORTS" in micro_labels:
        return "DISPUTED"
    elif "REFUTES" in micro_labels:
        return "REFUTES"
    elif "SUPPORTS" in micro_labels:
        return "SUPPORTS"
    else:
        return "NOT_ENOUGH_INFO"

def get_macro_labels_from_micro(micro_labels):
  macro_label_predictions=[]
  for i in range(0, len(micro_labels), 5):
    micro_labels_for_claim = micro_labels[i:i+5]
    macro_label_prediction = get_macro_label(micro_labels_for_claim)
    macro_label_predictions.append(macro_label_prediction)
  return macro_label_predictions

#with open('climate_fever/micro_label_predictions.json', 'r', encoding='utf-8') as f:
#      micro_label_predictions = json.load(f)
#      number_of_predictions_so_far = len(micro_label_predictions)

#print(number_of_predictions_so_far)

#no_errors = len([pred for pred in micro_label_predictions if pred=="ERROR"])
#print(no_errors)


with open('climate_fever/test_macro_labels.json', 'r', encoding='utf-8') as f:
    gold_macro_labels = json.load(f)

with open('climate_fever/test_micro_labels.json', 'r', encoding='utf-8') as f:
    gold_micro_labels = json.load(f)
    gold_micro_labels = [item for sublist in gold_micro_labels for item in sublist]

test = get_macro_labels_from_micro(gold_micro_labels)


for gold, test_label in zip(gold_macro_labels, test):
  if gold!=test_label:
    print((gold,test_label))