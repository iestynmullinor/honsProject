
import pickle
from sklearn.neighbors import KDTree
import time

# CREATES KD-TREE MODEL USING EUCLIDEAN DISTANCE

# Load sentence embeddings from embeddings.pkl

def create_kdtree(embeddings):

    # Build KD-Tree from sentence embeddings
    start = time.time()
    tree = KDTree(embeddings)
    end = time.time()
    print("Time taken to build KD-Tree: ", end - start)

    # Save KD-Tree as tree.pkl
    with open('sentence_similarity/tree.pkl', 'wb') as f:
        pickle.dump(tree, f)

if __name__ == "__main__":
    # Load the embeddings.pkl file
    with open('sentence_similarity/embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    create_kdtree(embeddings)