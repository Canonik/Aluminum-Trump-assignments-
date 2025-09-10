import torch, random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
import torch, time
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


dataset = load_dataset("json", data_files="trump_phrases.json")

MODEL_NAME = "google/flan-t5-small"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(base_model, lora_config)
model.train()

def format_example(example):
    phrase = example["text"]
    inputs = tokenizer(
        f"Trump's position:" + phrase,
        truncation=True,
        padding="max_length",
        max_length=200
    )
    targets = tokenizer(
        f"The response Trump used to state its position:" + phrase,
        truncation=True,
        padding="max_length",
        max_length=200
    )
    labels = [tid if tid != tokenizer.pad_token_id else -100 for tid in targets["input_ids"]]

    return {
        "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


train_data = dataset["train"].map(format_example)

eval_data = None


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-3,
    num_train_epochs=1,
    fp16=True,
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    predict_with_generate=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)


start_time = time.time()
trainer.train()
end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f}s")



model.save_pretrained("/content/neutralizer-lora-flan-t5-small")
tokenizer.save_pretrained("/content/neutralizer-lora-flan-t5-small")



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

for text in sample_texts:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    repetition_penalty=2.0,
    pad_token_id=tokenizer.eos_token_id
)
)
    print("Query:", text)
    print("Answer:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("------")

