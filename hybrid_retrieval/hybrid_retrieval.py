from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize  # You can replace this with your preferred tokenizer
import json
from nltk.corpus import stopwords
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

model = SentenceTransformer("all-mpnet-base-v2")

LLAMBDA = 0.2
NUMBER_AFTER_DECIMAL = str(LLAMBDA).split(".")[1]


with open ("sentence_similarity/data/sentence_section_pairs.json", "r", encoding="utf-8") as f:
    SENTENCE_SECTION_PAIRS = json.load(f)
    evidence_sentences = [pair[0] for pair in SENTENCE_SECTION_PAIRS]

def tokenize(text):
    # Return a list of tokens
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

with open('climate_fever/claims.json', 'r', encoding='utf-8') as f:
    CLIMATE_FEVER_CLAIMS = json.load(f)[:50]


# find the bm25 score for every sentence given a query
def get_bm25_scores(query):
    # Tokenize the query and evidence sentences
    tokenized_query = tokenize(query)
    tokenized_sentences = [tokenize(sentence) for sentence in evidence_sentences]

    # Create a BM25 object
    bm25 = BM25Okapi(tokenized_sentences)

    # Calculate BM25 scores for the query
    scores = bm25.get_scores(tokenized_query)

    # Apply min-max normalization to scale scores to [-1, 1] ( to be used in hybrid retrieval, as cosine similarity scores are in [-1, 1])
    min_score, max_score = min(scores), max(scores)
    # Avoid division by zero if all scores are the same
    if min_score == max_score:
        normalized_scores = [-1 if score < 0 else 1 for score in scores]  # Or use [0] * len(scores) if you prefer all scores to be 0 when they are all equal
    else:
        normalized_scores = [-1 + 2 * ((score - min_score) / (max_score - min_score)) for score in scores]

    return normalized_scores

def cosine_similarity(claim_embedding, evidence_embedding):
    # Normalize the embeddings to unit vectors
    claim_norm = np.linalg.norm(claim_embedding)
    evidence_norm = np.linalg.norm(evidence_embedding)
    
    # Avoid division by zero
    if claim_norm == 0 or evidence_norm == 0:
        return 0.0
    
    # Compute cosine similarity as dot product divided by norms (lengths)
    similarity = np.dot(claim_embedding, evidence_embedding) / (claim_norm * evidence_norm)
    return similarity

# find the dense retrieval results for a given query
def get_dense_retrieval_results(query):
    embeddings_directory = "sentence_similarity/model_evaluation/model_embeddings/MODEL_all-mpnet-base-v2_EMBEDDINGS.pkl"
    with open(embeddings_directory, "rb") as f:
        embeddings = pickle.load(f)

        if len(embeddings) != len(evidence_sentences):
            raise ValueError("The number of evidence sentences and embeddings do not match")

        query_embedding = model.encode(query, show_progress_bar=False, normalize_embeddings=True)
        
        # Calculate the cosine similarity between the query and every evidence sentences
        similarities = [cosine_similarity(query_embedding, evidence_embedding) for evidence_embedding in embeddings]

    return similarities

# find the hybrid retrieval results for query
def hybrid_retrieval(query):
    bm25_scores = get_bm25_scores(query)
    dense_scores = get_dense_retrieval_results(query)
        
        # Combine the BM25 and dense retrieval scores
    hybrid_scores = [(1-LLAMBDA) * bm25 + LLAMBDA * dense for bm25, dense in zip(bm25_scores, dense_scores)]

    return hybrid_scores

def get_k_nearest_for_claim(claim):
    scores = hybrid_retrieval(claim)
    # find the index of the top 5 scores
    top_3_indices = np.argsort(scores)[-3:]
    # find the top 5 evidence sentences
    top_3_sentences = [evidence_sentences[i] for i in top_3_indices]
    return top_3_sentences


# returns a dictionary of claims and their k (3) nearest neighbours
def get_k_nearest_for_all_claims(claims):
    k_nearest_for_all_claims = {}
    for  claim in tqdm(claims):
        k_nearest_for_all_claims[claim] = get_k_nearest_for_claim(claim)
    return k_nearest_for_all_claims

if __name__=="__main__":
    claims = CLIMATE_FEVER_CLAIMS
    k_nearest_for_all_claims = get_k_nearest_for_all_claims(claims)
    k_nearest_for_all_claims_str = json.dumps(k_nearest_for_all_claims, indent=4, ensure_ascii=False)
    with open(f'sentence_similarity/model_evaluation/model_relevance_evaluation/HYBRID_{NUMBER_AFTER_DECIMAL}_k_nearest_for_all_claims.txt', 'w', encoding='utf-8') as f:
        f.write(k_nearest_for_all_claims_str)
    



