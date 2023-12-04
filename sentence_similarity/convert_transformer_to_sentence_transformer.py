from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models

# Replace 'bert-base-uncased' with the name of the Hugging Face Transformer model you want to use
#model_name = 'climatebert/distilroberta-base-climate-f'
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name)

# Define the Sentence Transformer model with the loaded Hugging Face model as the encoder
#sentence_transformer_model = SentenceTransformer(modules=[models.Transformer(model_name)])

#loaded_sentence_transformer_model.save('climatebert_converted_to_sentence_transformer')

md = SentenceTransformer('climatebert_converted_to_sentence_transformer')

output = md.encode(['hello world', 'hello world'])