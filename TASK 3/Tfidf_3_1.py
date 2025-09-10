import os
import tiktoken
import json
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity


FILENAME= "trump_phrases.json"
TOKENIZER= "gpt-4o"

# Uses tiktoken tokenization
def text_tokenization(text): 
    encoder = tiktoken.encoding_for_model(TOKENIZER)
    tokenized = []
    for t in text:
        token_ids = encoder.encode(t)
        tokens = [str(tok) for tok in token_ids]
        tokenized.append(tokens)
    return tokenized

# Computes tf-idf for tokenized corpus and outpust similarity matrix for query indexing
def calculate_tf_idf(corpus):
    global model
    model = TfidfModel(corpus)
    vectors = []
    for c in corpus:
        vectors.append(model[c])
    return MatrixSimilarity(vectors)

def main():
    # Reads and parses JSON
    with open(FILENAME, encoding="utf-8") as file:
        phrases = []
        data = json.load(file)
        phrases += data 
    
    tokens = text_tokenization(phrases)
    
    # Creates token-id dictiornary
    dictionary = Dictionary(tokens)
    
    # Creates BoW
    corpus = [dictionary.doc2bow(token) for token in tokens]
    
    index = calculate_tf_idf(corpus)
    
    # Tokenizes query
    query = input("Phrase you want to search: ")
    query_tokens = text_tokenization([query])[0]
    
    # Creates query BoW in respect to the original dictionary
    query_corpus = dictionary.doc2bow(query_tokens)
    
    query_index = model[query_corpus]
    
    for doc_position, doc_score in sorted(enumerate(index[query_index]), key=lambda item: -item[1]):
        if doc_score > 0.10:
            print(f"{doc_score:.3f}", phrases[doc_position])

if __name__ == "__main__":
    main()