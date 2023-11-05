
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import pickle
from sklearn.neighbors import KDTree

# model being used for sentence embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# load sentnces
with open('sentence_similarity/sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

# load kd tree
with open('sentence_similarity/tree.pkl', 'rb') as f:
    tree = pickle.load(f)

# get input sentence from command line
input_sentence = input("Enter sentence: ")

# get sentence embeddings of input sentence
input_embedding = model.encode(input_sentence)

start = time.time()
# find 5 nearest neighbors to input sentence in the kd tree
distances, indices = tree.query([input_embedding], k=5)
end = time.time()
print("Time taken to find 5 nearest neighbors: ", end - start)

# print the 5 nearest neighbors
for index in indices[0]:
    print(sentences[index])


