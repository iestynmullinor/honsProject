from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import time
from sklearn.neighbors import KDTree
import compare_query
from model import MODEL as model
import pickle

reports = {'synthesis' : 'Synthesis Report', 
           'wg1' : 'Working Group 1 - Climate Change 2021: The Physical Science Basis',
           'wg2' : 'Working Group 2 - Climate Change 2022: Impacts, Adaptation and Vulnerability',
           'wg3' : 'Working Group 3 - Climate Change 2022: Mitigation of Climate Change'}

# Load sentence_section_pairs.pkl
with open('sentence_similarity/data/sentence_section_pairs.pkl', 'rb') as f:
    sentence_section_pairs = pickle.load(f)

# Access the loaded data
sentences = [pair[0] for pair in sentence_section_pairs]
sections = [pair[1] for pair in sentence_section_pairs]

def format_section(section):
    split_section = section.split('/')[1:]
    document_encoded = split_section[0]
    title = split_section[-1]
    document = reports[document_encoded]
    formatted_section = f"REPORT: {document}\nSECTION TITLE: {title}\n"
    return formatted_section



# finds the 5 nearest neighbors to the input sentence using cosine similarity
def nearest_cosine(input_embedding):
        
    # find 5 nearest neighbors to input sentence
    distances, indices = nn.kneighbors([input_embedding])
        
    # print the 5 nearest neighbors
    for index in indices[0]:
        print("EVIDENCE:", sentences[index])
        print(format_section(sections[index]))
        print("------------------------------------------------------------------------------------------------------------------------")

# finds the 5 nearest neighbors to the input sentence using euclidean distance
def nearest_euclidean(input_embedding):
    
    # find 5 nearest neighbors to input sentence in the kd tree
    distances, indices = tree.query([input_embedding], k=5)
    
    # print the 5 nearest neighbors
    for index in indices[0]:
        print("EVIDENCE:", sentences[index])
        print(format_section(sections[index]))
        print("------------------------------------------------------------------------------------------------------------------------")

if __name__=="__main__":
    # get input sentence from command line

    with open('sentence_similarity/nn_cosine.pkl', 'rb') as f:
        nn = pickle.load(f)

    with open('sentence_similarity/tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    
    while True:
        input_sentence = input("Enter sentence: ")
        
        # get sentence embeddings of input sentence
        input_embedding = model.encode(input_sentence)
        
        # only including this one for now as they all return same values every time

        print("\n")
        print("Potential Evidence for sentence: ", input_sentence)
        start = time.time()
        # find 5 nearest neighbors to input sentence using cosine similarity
        nearest_cosine(input_embedding)
        end = time.time()
        print("Time taken to find 5 nearest neighbors using cosine similarity: ", end - start)
        print("\n")



        #start = time.time()
        #find 5 nearest neighbors to input sentence using euclidean distance
        #print("Nearest neighbors using euclidean distance: ")
        #nearest_euclidean(input_embedding)
        #end = time.time()
        #print("Time taken to find 5 nearest neighbors using euclidean distance: ", end - start)

        #print("\n")

        #embeddings = pickle.load(open('sentence_similarity/embeddings.pkl', 'rb'))
        #sentences = pickle.load(open('sentence_similarity/sentences.pkl', 'rb'))
        #start = time.time()
        ## find 5 nearest neighbors to input sentence using iteration over list
        #print("Nearest neighbors using iteration over list: ")
        #compare_query.find_similar(input_sentence, embeddings, sentences)
        #end = time.time()
        #print("Time taken to find 5 nearest neighbors using iteration over list: ", end - start)

        #print("\n")