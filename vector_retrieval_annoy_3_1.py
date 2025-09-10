import os
import json
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer, util
np.random.seed(1234)

FILENAME= "trump_phrases.json"
NEIGHBORS = 3
METRIC = "angular"

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

    database = AnnoyIndex(dim, METRIC)
    for i, phrase in enumerate(phrases_embeddings):
        database.add_item(i, phrase.tolist())
    
    database.build(10) # 10 trees

    query = input("Phrase you want to search: ")
    query_embeddings = model.encode([query], convert_to_numpy=True).astype("float32")[0].tolist()

    ids, distances = database.get_nns_by_vector(query_embeddings, NEIGHBORS, include_distances=True)
    for idx, dist in zip(ids, distances):
        print(f"{phrases[idx]}  (distance={dist:.4f})")



if __name__ == "__main__":
    main()