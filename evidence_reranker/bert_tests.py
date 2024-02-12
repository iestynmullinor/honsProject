from transformers import AutoTokenizer, RobertaTokenizerFast
import pandas as pd
from datasets import load_dataset


#tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
# sentences_a = ["this is a sentence1", "this is a sentence2"]
# sentences_b = ["this is also a sentence1", "this is also a sentence2"]

# encoded_input = tokenizer(sentences_a, sentences_b, padding="max_length", truncation=True)

# print(tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0]))
# print(tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][1]))

dataset = load_dataset("iestynmullinor/climate_fever_reranker_training")
exit()

model_id = "climatebert/distilroberta-base-climate-f"


# relace the value with your model: ex <hugging-face-user>/<model-name>
repository_id = "iestynmullinor/climatebert-rerank-fever"

training_data_path = "evidence_reranker/training_data/climate_fever_reranker_training.csv"

#train_data = pd.read_csv(training_data_path)

train_data = load_dataset('csv', data_files=training_data_path)


tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

def tokenize(batch):
    claims = list(map(str, batch["claim"]))
    evidences = list(map(str, batch["evidence"]))
    tokenized_inputs = tokenizer(claims, evidences, padding=True, truncation=True, max_length=256)
    return tokenized_inputs

train_data= train_data.map(tokenize, batched=True, batch_size=len(train_data))

train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

