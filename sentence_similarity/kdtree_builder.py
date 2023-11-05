
import pickle
from sklearn.neighbors import KDTree
import time

# Load sentence embeddings from embeddings.pkl
with open('sentence_similarity/embeddings.pkl', 'rb') as f:
    start = time.time()
    embeddings = pickle.load(f)
    end = time.time()
    print("Time taken to load embeddings: ", end - start)

# Build KD-Tree from sentence embeddings
start = time.time()
tree = KDTree(embeddings)
end = time.time()
print("Time taken to build KD-Tree: ", end - start)

# Save KD-Tree as tree.pkl
with open('tree.pkl', 'wb') as f:
    pickle.dump(tree, f)
