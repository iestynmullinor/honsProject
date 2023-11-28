import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import pickle
from sklearn.neighbors import KDTree
from model import MODEL as model
import nn_cosine_builder
import kdtree_builder


# returns the sentence embedding for all sentences
def get_sentence_embeddings(sentences):
    return model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)

if __name__ == "__main__":
    print("Reading Data...")
    # Load the sentence_section_pairs.pkl file
    with open('sentence_similarity/data/sentence_section_pairs.pkl', 'rb') as f:
        sentence_section_pairs = pickle.load(f)

    sentences = [pair[0] for pair in sentence_section_pairs]

    print("Number of sentences: ", len(sentences))

    print("Generating sentence embeddings...")
    start = time.time()
    embeddings = get_sentence_embeddings(sentences)
    end = time.time()
    print("Time taken to extract sentence embeddings: ", end - start)

    print("Saving embeddings...")
    with open('sentence_similarity/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    print("Embeddings now saved in sentence_similarity/embeddings.pkl")
    print("Generating KDTree...")
    kdtree_builder.create_kdtree(embeddings)

    print("Generating Nearest Neighbors model...")
    nn_cosine_builder.generate_nn(embeddings)

    