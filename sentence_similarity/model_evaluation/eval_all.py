import pickle
from sentence_transformers import SentenceTransformer
import json
from sklearn.neighbors import NearestNeighbors
import json
RADIUS = 0.45
SEARCH_METRICS = False
RELEVANCE_TEST = True
NO_CLAIMS = 50
K = 3

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

EMBEDDINGS_NAMES = ['MODEL_all-mpnet-base-v2_EMBEDDINGS.pkl', 
          'MODEL_distilroberta-base-climate-f_EMBEDDINGS.pkl', 
          'MODEL_all-MiniLM-L12-v2_EMBEDDINGS.pkl',
          'MODEL_all-roberta-large-v1_EMBEDDINGS.pkl',
          'MODEL_bge-base-en-v1.5_EMBEDDINGS.pkl',
          'MODEL_all-MiniLM-L6-v2_EMBEDDINGS.pkl',
          'MODEL_climate-fever-msmarco-distilbert-gpl_EMBEDDINGS.pkl', #sentence transformers version of climate-bert
          'MODEL_bpr-gpl-climate-fever-base-msmarco-distilbert-tas-b_EMBEDDINGS.pkl',
          'MODEL_e5-large-v2_EMBEDDINGS.pkl',
          'MODEL_gte-large_EMBEDDINGS.pkl']

# all dummy claims are from https://huggingface.co/datasets/fever
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

    
with open('sentence_similarity/data/sentence_section_pairs.json', 'r', encoding='utf-8') as f:
    SENTENCE_SECTION_PAIRS = json.load(f)
SENTENCES = [pair[0] for pair in SENTENCE_SECTION_PAIRS]

# we find the "dummy" metrics for a model and add them to radius metrics, which evaluate how often it finds evidence for sentences unrelated to climate change
def get_dummy_metrics(nn, dummy_embeddings):
    dummy_metrics = {'no of dummy claims with evidence': 0, 'total no of neighbours of dummy claims': 0}
    
    for dummy_embedding in dummy_embeddings:
        distances, indices = nn.radius_neighbors([dummy_embedding])
        no_of_neighbours = len(indices[0])
        if no_of_neighbours > 0:
            dummy_metrics['no of dummy claims with evidence'] += 1
        dummy_metrics['total no of neighbours of dummy claims'] += no_of_neighbours
    
    dummy_metrics['percentage of dummy claims with evidence'] = dummy_metrics['no of dummy claims with evidence'] / len(dummy_embeddings) * 100
    dummy_metrics['average no of neighbours of dummy claims'] = dummy_metrics['total no of neighbours of dummy claims'] / len(dummy_embeddings)
    return dummy_metrics

# for a model, we find how many claims have no neighbours and the average number of neighbours per claim
def get_radius_metrics(nn, cf_embeddings):
    radius_metrics = {'no of claims with no neighbours': 0, 'total no of neighbours': 0}

    # for each claim, we find the number of neighbours it gets using this model
    for claim_embedding in cf_embeddings:
        distances, indices = nn.radius_neighbors([claim_embedding])
        no_of_neighbours = len(indices[0])
        if no_of_neighbours == 0:
            radius_metrics['no of claims with no neighbours'] += 1
        radius_metrics['total no of neighbours'] += no_of_neighbours

    radius_metrics['percentage of claims with no neighbours'] = radius_metrics['no of claims with no neighbours'] / len(cf_embeddings) * 100
    radius_metrics['average no of neighbours'] = radius_metrics['total no of neighbours'] / len(cf_embeddings)
    return radius_metrics

# gets the k (5) nearest neighbours for a claim
def get_k_nearest_for_claim(nn, claim_embedding):
    distances, indices = nn.kneighbors([claim_embedding])
    k_nearest = []
    for index in indices[0]:
        k_nearest.append(SENTENCES[index])
    return k_nearest

# returns a dictionary of claims and their k (5) nearest neighbours
def get_k_nearest_for_all_claims(nn, claims, cf_embeddings):
    k_nearest_for_all_claims = {}
    for claim, claim_embedding in zip(claims, cf_embeddings):
        k_nearest_for_all_claims[claim] = get_k_nearest_for_claim(nn, claim_embedding)
    return k_nearest_for_all_claims


# generates nearest neighbours model for a model
def generate_nn(embeddings, n_neighbours=K, radius=RADIUS):
    print("Generating Nearest Neighbors model...")
    # Create a NearestNeighbors instance
    nn = NearestNeighbors(n_neighbors=n_neighbours, metric='cosine', radius=radius)
    # Fit the model to your data
    nn.fit(embeddings)
    print("Done...")
    return nn

if __name__=="__main__":
    for model_name, embeddings_name, generic_name in zip(MODEL_NAMES, EMBEDDINGS_NAMES, MODEL_NAMES_WITHOUT_DIR):

        print(f"Generating metrics for {model_name}")
        with open(f'sentence_similarity/model_evaluation/model_embeddings/{embeddings_name}', 'rb') as f:
            embeddings = pickle.load(f)

        
        with open(f'sentence_similarity/model_evaluation/climate_fever_embeddings/MODEL_{generic_name}_CLIMATE_FEVER_EMBEDDINGS.pkl', 'rb') as f:
            cf_embeddings = pickle.load(f)

        with open(f'sentence_similarity/model_evaluation/dummy_embeddings/MODEL_{generic_name}_DUMMY_EMBEDDINGS.pkl', 'rb') as f:
            dummy_embeddings = pickle.load(f)

        nn = generate_nn(embeddings)

        if SEARCH_METRICS:
            print("Generating metrics...")
            radius_metrics = get_radius_metrics(nn, cf_embeddings)
            dummy_metrics = get_dummy_metrics(nn, dummy_embeddings)

            search_metrics = {**radius_metrics, **dummy_metrics}

            # Write radius metrics to a text file        
            search_metrics_str = json.dumps(search_metrics, indent=4)
            with open(f'sentence_similarity/model_evaluation/model_search_evaluation/{generic_name}_search_metrics.txt', 'w') as f:
                f.write(search_metrics_str)

            print("Done...")

        if RELEVANCE_TEST:
            print(f"Generating {K} nearest neighbours for first {NO_CLAIMS} claims...")
            # writes the 5 nearest neighbours for first 10 claims to a text file for manual evaluation

            k_nearest_for_all_claims = get_k_nearest_for_all_claims(nn, CLIMATE_FEVER_CLAIMS[:NO_CLAIMS], cf_embeddings[:NO_CLAIMS])
            k_nearest_for_all_claims_str = json.dumps(k_nearest_for_all_claims, indent=4, ensure_ascii=False)
            with open(f'sentence_similarity/model_evaluation/model_relevance_evaluation/{generic_name}_k_nearest_for_all_claims.txt', 'w', encoding='utf-8') as f:
                f.write(k_nearest_for_all_claims_str)

        
    
        
