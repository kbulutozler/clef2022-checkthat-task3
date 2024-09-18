import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, TrainingArguments

import pandas as pd
from huggingface_hub import login
import traceback

login(token="")

model_id = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=2,  # Adjust based on your classification task
    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id for the model
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_dir='./logs',
    logging_steps=100,
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

lora_config = LoraConfig(
    r=8,
    target_modules="all-linear",
    bias="none",
    task_type="SEQ_CLS",  # Change to sequence classification
)

# Load and preprocess the data
df = pd.read_csv('/workspace/clef2022-checkthat-task3/data/preprocessed/train.csv')
print(df.head())
# Check average sequence length
tokenized_lengths = df['title_text'].apply(lambda x: len(tokenizer.encode(x)))
average_length = tokenized_lengths.mean()
max_length = tokenized_lengths.max()

print(f"Average sequence length: {average_length:.2f}")
print(f"Maximum sequence length: {max_length}")
print("wow")

# Create label2id dictionary
unique_labels = df['label'].unique()
label2id = {label: id for id, label in enumerate(unique_labels)}
id2label = {id: label for label, id in label2id.items()}

print("Label to ID mapping:", label2id)

def preprocess_function(examples):
    # Assuming examples is now a batch of data
    result = tokenizer(examples["title_text"], truncation=True, padding="max_length", max_length=1024)
    result["labels"] = [label2id[label] for label in examples["label"]]
    return result

# Update the batch_tokenize function
def batch_tokenize(df, batch_size=32):
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        yield preprocess_function(batch.to_dict(orient="list"))

dataset = Dataset.from_dict(
    {k: sum((batch[k] for batch in batch_tokenize(df)), [])
     for k in next(batch_tokenize(df)).keys()}
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()
