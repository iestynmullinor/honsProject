import pickle
from transformers import AutoTokenizer, AutoModel
import json
from sklearn.neighbors import NearestNeighbors
import json

K=3
RADIUS=1
NO_CLAIMS=50

MODEL_NAMES = ['contriever',
               'contriever-msmarco']

with open('climate_fever/claims.json', 'r', encoding='utf-8') as f:
    CLIMATE_FEVER_CLAIMS = json.load(f)

with open('sentence_similarity/data/sentence_section_pairs.json', 'r', encoding='utf-8') as f:
    SENTENCE_SECTION_PAIRS = json.load(f)
SENTENCES = [pair[0] for pair in SENTENCE_SECTION_PAIRS]

with open('climate_fever/claims.json', 'r', encoding='utf-8') as f:
    CLIMATE_FEVER_CLAIMS = json.load(f)

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
    for model_name in MODEL_NAMES:

        print(f"Generating metrics for {model_name}")
        with open(f'contriever_&_climatebert/contriever_embeddings/MODEL_{model_name}_EMBEDDINGS.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        nn = generate_nn(embeddings)

        with open(f'contriever_&_climatebert/contriever_cf_embeddings/MODEL_{model_name}_CF_EMBEDDINGS.pkl', 'rb') as f:
            cf_embeddings = pickle.load(f)


        print(f"Generating {K} nearest neighbours for first {NO_CLAIMS} claims...")
        # writes the 5 nearest neighbours for first 10 claims to a text file for manual evaluation

        k_nearest_for_all_claims = get_k_nearest_for_all_claims(nn, CLIMATE_FEVER_CLAIMS[:NO_CLAIMS], cf_embeddings[:NO_CLAIMS])
        k_nearest_for_all_claims_str = json.dumps(k_nearest_for_all_claims, indent=4, ensure_ascii=False)
        with open(f'sentence_similarity/model_evaluation/model_relevance_evaluation/{model_name}_k_nearest_for_all_claims.txt', 'w', encoding='utf-8') as f:
            f.write(k_nearest_for_all_claims_str)
