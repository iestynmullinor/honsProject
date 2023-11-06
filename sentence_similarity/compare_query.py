import numpy as np
from sentence_transformers import SentenceTransformer
import time
import pickle
from model import MODEL as model

# calculates the cosine similarity between two sentences 
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

#create a list of cosine similarities between a query and all sentences
def calculate_similarity(query_embedding, embeddings):
    similarities = []
    for embedding in embeddings:
        similarities.append(cosine(query_embedding, embedding))
    return similarities

# returns the index of the top 5 sentences with the highest cosine similarity
def find_index_of_top_5_sentences(similarities):
    top_5 = []
    for i in range(5):
        top_5.append(similarities.index(max(similarities)))
        similarities[similarities.index(max(similarities))] = 0
    return top_5

# returns the top 5 sentences with the highest cosine similarity
def get_top_5_sentences(sentences, top_5):
    top_5_sentences = []
    for i in top_5:
        top_5_sentences.append(sentences[i])
    return top_5_sentences


def find_similar(query, embeddings, sentences):
    embeddings = pickle.load(open('sentence_similarity/embeddings.pkl', 'rb'))
    print("number of embeddings: ", len(embeddings))
    sentences = pickle.load(open('sentence_similarity/sentences.pkl', 'rb'))
    start = time.time()
    query_embedding = model.encode([query])[0]
    similarities = calculate_similarity(query_embedding, embeddings)    
    top_5_index = find_index_of_top_5_sentences(similarities)
    top_5_sents = get_top_5_sentences(sentences, top_5_index)
    end = time.time()
    for sent in top_5_sents:
        print(sent)
    print("Time taken to find top 5 sentences: ", end - start)