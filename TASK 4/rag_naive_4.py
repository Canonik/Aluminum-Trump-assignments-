
import torch, random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
import torch, time
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

dataset = load_dataset("json", data_files="trump_phrases.json")
phrases = list(dataset["train"]["text"])

MODEL_NAME = "google/flan-t5-large"

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="trump_phrases")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedder.encode(phrases).tolist()
ids = [str(i) for i in range(len(phrases))]

collection.add(documents=phrases, embeddings=embeddings, ids=ids)

sample_texts = [
    "What about COVID-19?",
    "Summarize Trump’s statements on the COVID-19 pandemic, focusing on his views on the government’s response, the role of China, and measures he promoted.",
    "How’s the economy?",
    "Summarize Trump’s position on the U.S. economy, including his claims about growth, unemployment, and stock market performance, with references to transcript passages.",
    "What about tariffs?",
    "Summarize Trump’s stance on tariffs, especially toward China, including his stated goals, justifications, and expected outcomes, citing transcript passages when possible.",
    "What about immigration?",
    "Summarize Trump’s position on immigration policy, highlighting his views on border security, the wall, and restrictions on entry, with supporting quotes.",
    "What about national security?",
    "Summarize Trump’s views on national security, focusing on military strength, terrorism, and foreign policy challenges, and reference transcript evidence when relevant."
]

def rag_query(query):
   
    query_emb = embedder.encode([query]).tolist()

    results = collection.query(query_embeddings=query_emb, n_results=5)
    retrieved_docs = " ".join(results["documents"][0])  

    prompt = f"You are Trump, answer using his phrases to the Question:\n{retrieved_docs}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    outputs = base_model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.pad_token_id)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for q in sample_texts:
    print("Query:", q )
    print("Answer:", rag_query(q))
    print("------")

