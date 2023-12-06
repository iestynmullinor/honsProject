# Load model directly
from transformers import AutoTokenizer, SentencePairClassifier

tokenizer = AutoTokenizer.from_pretrained("mwong/ernie-v2-fever-evidence-related")
model = SentencePairClassifier.from_pretrained("mwong/ernie-v2-fever-evidence-related")

print(model("James is a man. James is a human man."))
print(model("James is a man. The sky is turning red."))
