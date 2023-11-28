from sentence_transformers import SentenceTransformer
import pickle

MODEL_NAMES = ['all-mpnet-base-v2', 
          'climatebert/distilroberta-base-climate-f', 
          'all-MiniLM-L12-v2',
          'sentence-transformers/all-roberta-large-v1',
          'BAAI/bge-base-en-v1.5',
          'sentence-transformers/all-MiniLM-L6-v2',
          'GPL/climate-fever-msmarco-distilbert-gpl', #sentence transformers version of climate-bert
          'income/bpr-gpl-climate-fever-base-msmarco-distilbert-tas-b',
          'intfloat/e5-large-v2',
          'thenlper/gte-large']

MODEL_NAMES_WITHOUT_DIR = ['all-mpnet-base-v2', 
          'distilroberta-base-climate-f', 
          'all-MiniLM-L12-v2',
          'all-roberta-large-v1',
          'bge-base-en-v1.5',
          'all-MiniLM-L6-v2',
          'climate-fever-msmarco-distilbert-gpl', #sentence transformers version of climate-bert
          'bpr-gpl-climate-fever-base-msmarco-distilbert-tas-b',
          'e5-large-v2',
          'gte-large']


def get_sentences():
    print("Reading Data...")
    # Load the sentence_section_pairs.pkl file
    with open('sentence_similarity/data/sentence_section_pairs.pkl', 'rb') as f:
        sentence_section_pairs = pickle.load(f)
    sentences = [pair[0] for pair in sentence_section_pairs]
    return sentences

def get_sentence_embeddings(sentences, model):
    print("Generating sentence embeddings...")
    return model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)

def get_climate_fever_claims():
    with open('climate_fever/claims.pkl', 'rb') as f:
        claims = pickle.load(f)
    return claims

if __name__=="__main__":
    sentences = get_sentences()
    for model_name in MODEL_NAMES:
        index = MODEL_NAMES.index(model_name)
        print(f"Generating embeddings for {model_name}")
        print(f"this is model number {index}")
        model = SentenceTransformer(model_name)
        embeddings = get_sentence_embeddings(sentences, model)
        with open(f'sentence_similarity/model_evaluation/model_embeddings/MODEL_{MODEL_NAMES_WITHOUT_DIR[index]}_EMBEDDINGS.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        
        
        
        

