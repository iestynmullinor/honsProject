from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import time

# CREATES NEAREST NEIGHBORS MODEL USING COSINE SIMILARITY


# load sentences
with open('sentence_similarity/sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

# load embeddings
with open('sentence_similarity/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Create a NearestNeighbors instance
nn = NearestNeighbors(n_neighbors=5, metric='cosine')

# Fit the model to your data
nn.fit(embeddings)

# save the model
with open('sentence_similarity/nn_cosine.pkl', 'wb') as f:
    pickle.dump(nn, f)