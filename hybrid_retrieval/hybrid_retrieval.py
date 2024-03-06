from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize  # You can replace this with your preferred tokenizer
import json
from nltk.corpus import stopwords
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-mpnet-base-v2")


with open ("sentence_similarity/data/sentence_section_pairs.json", "r", encoding="utf-8") as f:
    SENTENCE_SECTION_PAIRS = json.load(f)
    evidence_sentences = [pair[0] for pair in SENTENCE_SECTION_PAIRS]

def tokenize(text):
    # Return a list of tokens
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


# find the bm25 score for every sentence given a query
def get_bm25_scores(query):
    # Tokenize the query and evidence sentences
    tokenized_query = tokenize(query)
    tokenized_sentences = [tokenize(sentence) for sentence in evidence_sentences]

    # Create a BM25 object
    bm25 = BM25Okapi(tokenized_sentences)

    # Calculate BM25 scores for the query
    scores = bm25.get_scores(tokenized_query)

    return scores

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

        query_embedding = model.encode(query, show_progress_bar=True, normalize_embeddings=True)
        
        # Calculate the cosine similarity between the query and every evidence sentences
        similarities = [cosine_similarity(query_embedding, evidence_embedding) for evidence_embedding in embeddings]

    return similarities

# find the hybrid retrieval results for query
def hybrid_retrieval(query):
    bm25_scores = get_bm25_scores(query)
    dense_scores = get_dense_retrieval_results(query)
        
        # Combine the BM25 and dense retrieval scores
    hybrid_scores = [0.2 * bm25 + 0.8 * dense for bm25, dense in zip(bm25_scores, dense_scores)]

    return hybrid_scores

if __name__ == "__main__":
    queries = "What is the incubation period of COVID-19?"
    scores = hybrid_retrieval(queries)
    # find the index of the top 5 scores
    top_5_indices = np.argsort(scores)[-5:]
    # find the top 5 evidence sentences
    top_5_sentences = [evidence_sentences[i] for i in top_5_indices]
    print(top_5_sentences)


