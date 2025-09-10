import os
import tiktoken
import json
from rank_bm25 import BM25Okapi

FILENAME= "trump_phrases.json"
DOCS= 5

# Super simple tokenizer
def text_tokenization(texts): 
    return [t.lower().split() for t in texts]

def main():
    # Reads and parses JSON
    with open(FILENAME, encoding="utf-8") as file:
        phrases = []
        data = json.load(file)
        phrases = list(data)
    
    tokens = text_tokenization(phrases)
    bm25 = BM25Okapi(tokens)

    query = input("Phrase you want to search: ")
    query_tokens = text_tokenization([query])[0]
    
    scores = bm25.get_scores(query_tokens)
    phrases = bm25.get_top_n(query_tokens, phrases, n=DOCS)
    for score, doc in zip(sorted(scores, reverse=True), phrases):
        print(f"{score:.3f} " + doc)


if __name__== "__main__":
    main()