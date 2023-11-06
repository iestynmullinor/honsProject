from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import time
from sklearn.neighbors import KDTree
import compare_query

# model being used for sentence embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# load sentences#
with open('sentence_similarity/sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

# finds the 5 nearest neighbors to the input sentence using cosine similarity
def nearest_cosine(input_embedding):
    # load nn_cosine model
    with open('sentence_similarity/nn_cosine.pkl', 'rb') as f:
        nn = pickle.load(f)
    
    # find 5 nearest neighbors to input sentence
    distances, indices = nn.kneighbors([input_embedding])
    
    # print the 5 nearest neighbors
    for index in indices[0]:
        print(sentences[index])

# finds the 5 nearest neighbors to the input sentence using euclidean distance
def nearest_euclidean(input_embedding):
    # load tree model
    with open('sentence_similarity/tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    
    # find 5 nearest neighbors to input sentence in the kd tree
    distances, indices = tree.query([input_embedding], k=5)
    
    # print the 5 nearest neighbors
    for index in indices[0]:
        print(sentences[index])

if __name__=="__main__":
    # get input sentence from command line
    input_sentence = input("Enter sentence: ")
    
    # get sentence embeddings of input sentence
    input_embedding = model.encode(input_sentence)
    
    # only including this one for now as they all return same values every time

    start = time.time()
    # find 5 nearest neighbors to input sentence using cosine similarity
    print("Nearest neighbors using cosine similarity: ")
    nearest_cosine(input_embedding)
    end = time.time()
    print("Time taken to find 5 nearest neighbors using cosine similarity: ", end - start)

    #print("\n")

    #start = time.time()
    # find 5 nearest neighbors to input sentence using euclidean distance
    #print("Nearest neighbors using euclidean distance: ")
    #nearest_euclidean(input_embedding)
    #end = time.time()
    #print("Time taken to find 5 nearest neighbors using euclidean distance: ", end - start)

    #print("\n")

    #embeddings = pickle.load(open('sentence_similarity/embeddings.pkl', 'rb'))
    #sentences = pickle.load(open('sentence_similarity/sentences.pkl', 'rb'))
    #start = time.time()
    # find 5 nearest neighbors to input sentence using iteration over list
    #print("Nearest neighbors using iteration over list: ")
    #compare_query.find_similar(input_sentence, embeddings, sentences)
    #end = time.time()
    #print("Time taken to find 5 nearest neighbors using iteration over list: ", end - start)