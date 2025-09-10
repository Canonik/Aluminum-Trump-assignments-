import os
import json
import numpy as np
import faiss 
from sentence_transformers import SentenceTransformer, util
np.random.seed(1234)

FILENAME= "trump_phrases.json"
NEIGHBORS = 3

model_path = "ibm-granite/granite-embedding-english-r2"
model = SentenceTransformer(model_path)

def main():
    # Reads and parses JSON
    with open(FILENAME, encoding="utf-8") as file:
        phrases = []
        data = json.load(file)
        phrases = list(data)
    
    phrases_embeddings = model.encode(phrases, convert_to_numpy=True).astype("float32")
    dim = phrases_embeddings.shape[1]    
    index = faiss.IndexFlatL2(dim)   # build the index
    index.add(phrases_embeddings.astype("float32"))   # add vectors to the index
    
    query = input("Phrase you want to search: ")
    query_embeddings = model.encode([query], convert_to_numpy=True)

    D, I = index.search(query_embeddings, NEIGHBORS)
    print("\nTop matches:")
    for idx, dist in zip(I[0], D[0]):
        print(f"{phrases[idx]}  (distance={dist:.4f})")

if __name__ == "__main__":
    main()                