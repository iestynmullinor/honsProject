from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize  # You can replace this with your preferred tokenizer
import json
from nltk.corpus import stopwords


def tokenize(text):
    # Return a list of tokens
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def find_nearest_neighbors(query, evidence_sentences, k=5, k1=1.5, b=0.75):
    # Tokenize the query and evidence sentences
    tokenized_query = tokenize(query)
    tokenized_sentences = [tokenize(sentence) for sentence in evidence_sentences]

    # Create a BM25 object
    bm25 = BM25Okapi(tokenized_sentences,  k1=k1, b=b)

    # Calculate BM25 scores for the query
    scores = bm25.get_scores(tokenized_query)

    # Rank evidence sentences based on BM25 scores
    ranked_sentences = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

    # Retrieve the indices of the top k evidence sentences
    top_indices = [index for index, _ in ranked_sentences[:k]]

    # Get the actual sentences corresponding to the top indices
    nearest_neighbors = [(evidence_sentences[index], index) for index in top_indices]

    return nearest_neighbors

def get_evidence(query_claim):
    with open ("sentence_similarity/data/sentence_section_pairs.json", "r", encoding="utf-8") as f:
        SENTENCE_SECTION_PAIRS = json.load(f)
    evidence_sentences = [pair[0] for pair in SENTENCE_SECTION_PAIRS]


    # Adjust k1 and b as needed
    processed_evidence_sentences = [sentence.lower() for sentence in evidence_sentences]
    nearest_neighbors = find_nearest_neighbors(query_claim, processed_evidence_sentences, k=3, k1=1.5, b=0.75)
    indexes = [index for _, index in nearest_neighbors]
    nearest_neighbors = [evidence_sentences[index] for index in indexes]

    return nearest_neighbors

if __name__ == "__main__":
    with open("climate_fever/claims.json", "r", encoding="utf-8") as f:
        claims = json.load(f)[:50]
    results_dict = {}
    for claim in claims:
        print("claim number:", claims.index(claim))
        query_claim = claim
        evidence_sentences = get_evidence(query_claim)
        results_dict[query_claim] = evidence_sentences
        result_json = json.dumps(results_dict, indent=4, ensure_ascii=False)
    with open("sentence_similarity/model_evaluation/model_relevance_evaluation/bm25_k_nearest_for_all_claims.txt", "w", encoding="utf-8") as f:
        f.write(result_json)
