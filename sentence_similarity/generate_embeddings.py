import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import pickle
from sklearn.neighbors import KDTree

# model being used for sentence embeddings

# all mpnet model 
model = SentenceTransformer('all-mpnet-base-v2')

# climatebert base model
#sbert_model = SentenceTransformer('climatebert/distilroberta-base-climate-f')

# climatebert evidence model (seems to not work)
#sbert_model = SentenceTransformer('mwong/climatebert-base-f-fever-evidence-related')

#extracts sentences from a file
def extract_sentences(path_to_file):
    with open(path_to_file, 'r') as f:
        text = f.read()
    sentences = sent_tokenize(text)
    return sentences

# returns the sentence embedding for all sentences
def get_sentence_embeddings(sentences):
    return model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)


FILE = "data_extraction/AR6_whole/ipcc_ar6.txt"

if __name__ == "__main__":
    print("Extracting sentences from file...")
    sentences = extract_sentences(FILE)
    print("Number of sentences: ", len(sentences))

    print("Saving sentences...")
    with open('sentence_similarity/sentences.pkl', 'wb') as f:
        pickle.dump(sentences, f)

    print("Extracting sentence embeddings...")
    start = time.time()
    embeddings = get_sentence_embeddings(sentences)
    end = time.time()

    print("Time taken to extract sentence embeddings: ", end - start)

    print("building kd tree...")
    start = time.time()
    # Build a KD-Tree from the normalized embeddings
    kdtree = KDTree(embeddings, metric='cosine')
    end = time.time()
    print("Time taken to build kd tree: ", end - start)

    print("Saving kdtree...")
    with open('sentence_similarity/kdtree.pkl', 'wb') as f:
        pickle.dump(kdtree, f)


    