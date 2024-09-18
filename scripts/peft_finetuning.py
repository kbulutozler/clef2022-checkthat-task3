import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  

import pandas as pd
from huggingface_hub import login

login(token="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")

model_id = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

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
    task_type="CAUSAL_LM",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="text",
)
