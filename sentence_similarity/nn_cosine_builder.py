from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import time

# CREATES NEAREST NEIGHBORS MODEL USING COSINE SIMILARITY



def generate_nn(embeddings, n_neighbours=10, radius=0.45):

    # Create a NearestNeighbors instance
    nn = NearestNeighbors(n_neighbors=n_neighbours, metric='cosine', radius=radius)

    # Fit the model to your data
    nn.fit(embeddings)

    # save the model
    with open('sentence_similarity/nn_cosine.pkl', 'wb') as f:
        pickle.dump(nn, f)

if __name__ == "__main__":
    # load embeddings
    with open('sentence_similarity/embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    start = time.time()
    generate_nn(embeddings)
    end = time.time()
    print("Time taken to build Nearest Neighbors model: ", end - start)