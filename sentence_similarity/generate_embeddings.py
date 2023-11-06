import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import pickle
from sklearn.neighbors import KDTree
from model import MODEL as model


# extracts sentences from a file that are at least 7 words long and removes duplicates
def extract_sentences(path_to_file):
    with open(path_to_file, 'r') as f:
        text = f.read()
    sentences = sent_tokenize(text)
    sentences = list(set(sentences))  # remove duplicates
    sentences = [s for s in sentences if len(word_tokenize(s)) >= 7]  # only include sentences with at least 7 words
    return sentences


# returns the sentence embedding for all sentences
def get_sentence_embeddings(sentences):
    return model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)


FILE = "data_extraction/AR6_whole/AR6_whole_cleaned.txt"

if __name__ == "__main__":
    print("Extracting sentences from file...")
    sentences = extract_sentences(FILE)
    print("Number of sentences: ", len(sentences))

    print("Saving sentences...")
    with open('sentence_similarity/sentences.pkl', 'wb') as f:
        pickle.dump(sentences, f)

    print("writing sentences to .txt file for inspection...")
    with open('sentence_similarity/sentences.txt', 'w') as f:
        for s in sentences:
            f.write(s + "\n")

    print("Generating sentence embeddings...")
    start = time.time()
    embeddings = get_sentence_embeddings(sentences)
    end = time.time()
    print("Time taken to extract sentence embeddings: ", end - start)

    print("Saving embeddings...")
    with open('sentence_similarity/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    print("Embeddings now saved in sentence_similarity/embeddings.pkl")
    print("Remember to run kdtree_builder.py to generate the KDTree and to run nn_cosine_builder.py to generate the NN cosine data structure")


    