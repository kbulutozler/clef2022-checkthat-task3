from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_scheduler
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics import F1Score, Accuracy
import os

def write_results(path, results_to_write, hpmeters):
    results_file = open(os.path.join(path, "all_results_on_eval.txt"), "w")
    for results_line in results_to_write:
        results_file.write(results_line + "\n")
    for k,v in hpmeters.items():
        results_file.write(k + ': ' + str(v) + "\n")
    results_file.close()
    
def tokenize(samples):
    return tokenizer(samples["text"], padding="max_length", truncation=True)

def prepare_dataset(dataset):
    tokenized_dataset = dataset.map(tokenize, batched=True)
    for col in tokenized_dataset.features.keys():
        if col != 'input_ids' and col != 'labels':
            tokenized_dataset = tokenized_dataset.remove_columns([col])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def clean_dataset(df):
    df = df.rename(columns={"title_text": "text", "label": "labels"})
    df.loc[df["labels"] == "partial", "labels"] = 1
    df.loc[df["labels"] == "other", "labels"] = 3
    df.loc[df["labels"] == "truth", "labels"] = 2
    df.loc[df["labels"] == "fake", "labels"] = 0
    return df

def compute_metrics(predictions, labels):
    results = {}
    f1_micro = F1Score(num_classes=4, average='micro').to(device)
    f1_macro = F1Score(num_classes=4, average='macro').to(device)
    accuracy = Accuracy(num_classes=4).to(device)
    results["f1_micro"] = f1_micro(predictions, labels).cpu().numpy().item()
    results["f1_macro"] = f1_macro(predictions, labels).cpu().numpy().item()
    results["f1_macro"] = accuracy(predictions, labels).cpu().numpy().item()
    return results

def train_eval_loop(model, train_dataloader, optimizer, scheduler, epochs, training_steps):
    progress_bar_train = tqdm(range(training_steps))
    results_to_write = []
    list_train_loss = []
    for epoch in range(epochs):
        model.train()
        predictions = []
        labels = []
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar_train.update(1)
            list_train_loss.append(loss.item())
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1)
            predictions.append(batch_predictions)
            labels.append(batch["labels"])
            
        
        predictions = torch.cat(tuple(predictions), 0)
        labels = torch.cat(tuple(labels), 0)
        results = compute_metrics(predictions, labels)
        print(f"epoch {epoch+1}:", results)
        results_line = "epoch: " + str(epoch+1) + " f1_micro: " + str(results["f1_micro"]) + " f1_macro: " + str(results["f1_macro"])+ " accuracy: " + str(results["accuracy"])
        results_to_write.append(results_line)
    return model, results_to_write, list_train_loss




base_model_name = "distilbert-base-uncased"
output_path = f"/app/output/{base_model_name}/"
try: 
    os.makedirs(output_path)
except FileExistsError:
    pass
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model_name)
print("tokenizer loaded")
dataset_path = "/app/data/preprocessed"

train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
train_df = clean_dataset(train_df)
train_dataset = Dataset.from_pandas(train_df)
tokenized_train_dataset = prepare_dataset(train_dataset)
tokenized_train_dataset = tokenized_train_dataset.shuffle(seed=42)
print("train dataset tokenized")

num_epochs = 1
learning_rate = 2e-5
batch_size = 2
hpmeters = {'num_epochs':num_epochs, 'learning_rate':learning_rate, 'batch_size':batch_size}
num_labels = 4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # set it to cpu during debugging
print(device)
#small_tokenized_train_dataset = tokenized_train_dataset.select(range(0,50))
train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size)


model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=base_model_name, 
                                                           num_labels=num_labels)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(name="linear", 
                             optimizer=optimizer, 
                             num_warmup_steps=0, 
                             num_training_steps=num_training_steps)

model, results_to_write, list_train_loss = train_eval_loop(model=model, 
                train_dataloader=train_dataloader, 
                optimizer=optimizer, 
                scheduler=lr_scheduler, 
                epochs=num_epochs, 
                training_steps=num_training_steps)

print("training is done") 
output_path = os.path.join(output_path, f"num_epochs_{num_epochs}-batch_size_{batch_size}-learning_rate_{learning_rate}")           
write_results(output_path, results_to_write, hpmeters)
print("results are written to: ", output_path)
model.save_pretrained(output_path)
print("model has been saved to: ", output_path)

