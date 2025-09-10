import os
import json
import time
import spacy
nlp = spacy.load("en_core_web_sm")

def data_parsing() -> list[str]:
    all_entries = []
    for filename in os.listdir("C:/Users/aless/Documents/AI_assignments/Dataset"):
        if filename.endswith(".json"):
            path = os.path.join("C:/Users/aless/Documents/AI_assignments/Dataset", filename)
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
                all_entries.append(data)

    trump_text = []
    for entry in all_entries:
        dialogue = entry.get("dialogue", [])
        for phrase in dialogue:
            if phrase.get("speaker") == "Donald Trump":
                trump_text.append(phrase.get("text").strip())
    return trump_text

def has_subject_predicate(doc, sentence: str) -> bool:
    has_subj = any(tok.dep_ in ("nsubj", "nsubjpass", "csubj") for tok in doc)
    has_verb = any(tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX") for tok in doc)
    has_noun = any(tok.pos_ in ("NOUN", "PROPN") for tok in doc)
    max_len = len(sentence.split()) <= 60
    ends_proper = sentence.strip().endswith(('.', '!', '?'))
    return has_subj and has_verb and has_noun and ends_proper and max_len


def main(): 
    trump_text = data_parsing()
    valid_text = []
    start_time = time.time()

    for doc, sentence in zip(nlp.pipe(trump_text, batch_size=100), trump_text):
        if has_subject_predicate(doc, sentence):
            valid_text.append(sentence)

    with open("filtered_trump_sentences.json", "w", encoding="utf-8") as f:
        json.dump(valid_text, f, ensure_ascii=False, indent=2)


    end_time = time.time()
    print(f"Total_time: {end_time-start_time:.2f}s")
    
if __name__ == "__main__":
    main()