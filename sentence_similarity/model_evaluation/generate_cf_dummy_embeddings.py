from sentence_transformers import SentenceTransformer
import pickle
import json

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


DUMMY_CLAIMS = ['Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.',
                'Roman Atwood is a content creator.'
                'The Challenge was a scripted show.',
                'Willie Nelson dropped out of college after three years.',
                'Led Zeppelin is a band.',
                'Peter Davison has been in a BBC show.',
                'Overwatch is a video game in which players gain cosmetic awards and it assigns players into two teams of six, each player with a style of play known as Offense, Defense, Tank, and Support.',
                'The Peloponnesian War was won by Sparta.',
                'Planet of the Apes (1968 film) had a script that underwent rewrites.',
                'Since July 2012, Pranab Mukherjee has been in office.',
                'Israel located west of China.',
                'Charles, Prince of Wales is patron of numerous other organizations.',
                'Another One Bites the Dust was written by a pianist.',
                'Adele has a song called Hello.',
                'Major League Soccer is banned in Australia.',
                'Lymphoma, the leading cause of death in the world, caused 305,000 deaths in 2012.',
                'We Found Love is a romance novel.',
                'Batman v Superman: Dawn of Justice is a television show.',
                'The Twilight Saga stars an actor born on a plane.',
                'Reese Witherspoon received an Oscar nomination.'
                ]

with open('climate_fever/claims.json', 'r', encoding='utf-8') as f:
    CLIMATE_FEVER_CLAIMS = json.load(f)

# generates climate_fever claims embeddings for a model
def generate_claim_dummy_embeddings(model_name):
    print(f"Generating embeddings for {model_name}")
    model = SentenceTransformer(model_name)
    cf_embeddings = model.encode(CLIMATE_FEVER_CLAIMS, show_progress_bar=True, normalize_embeddings=True)
    dummy_embeddings = model.encode(DUMMY_CLAIMS, show_progress_bar=True, normalize_embeddings=True)
    return cf_embeddings, dummy_embeddings

if __name__=="__main__":
    for model_name,generic_name in zip(MODEL_NAMES,MODEL_NAMES_WITHOUT_DIR):
        cf_embeddings, dummy_embeddings = generate_claim_dummy_embeddings(model_name)
        with open(f'sentence_similarity/model_evaluation/climate_fever_embeddings/MODEL_{generic_name}_CLIMATE_FEVER_EMBEDDINGS.pkl', 'wb') as f:
            pickle.dump(cf_embeddings, f)
        with open(f'sentence_similarity/model_evaluation/dummy_embeddings/MODEL_{generic_name}_DUMMY_EMBEDDINGS.pkl', 'wb') as f:
            pickle.dump(dummy_embeddings, f)