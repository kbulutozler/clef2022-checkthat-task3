from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os


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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # set it to cpu during debugging
batch_size=1
base_model_name = "distilbert-base-uncased" # or allenai/longformer-base-4096
model_path = os.path.join("/app/output", base_model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model_name)
print("tokenizer loaded")
dataset_path = "/app/data/preprocessed"



model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)

model.to(device)
dev_df = pd.read_csv(os.path.join(dataset_path, 'dev.csv'))
dev_df = clean_dataset(dev_df)
dev_dataset = Dataset.from_pandas(dev_df)
tokenized_dev_dataset = prepare_dataset(dev_dataset)
tokenized_dev_dataset = tokenized_dev_dataset.shuffle(seed=42)
eval_dataloader = DataLoader(tokenized_dev_dataset, batch_size=batch_size)


results = {0:{0:0, 1:0,2:0,3:0},
           1:{0:0, 1:0,2:0,3:0},
           2:{0:0, 1:0,2:0,3:0},
           3:{0:0, 1:0,2:0,3:0}
          }

label_names = {0:"fake", 1:"partially fake/truth", 2: "truth", 3:"other"}

for i, sample in enumerate(eval_dataloader):
    sample = {k: v.to(device) for k, v in sample.items()}
    #text = tokenizer.decode(sample['input_ids'].tolist()[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    with torch.no_grad():
        outputs = model(**sample)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    label = sample['labels'].item()
    results[label][prediction] += 1

for gold_label, predictions in results.items():
    total_count = 0
    for predicted_label, count in predictions.items():
        total_count+=count
    print("for the", total_count, "samples with gold label", label_names[gold_label], ":")
    for predicted_label, count in predictions.items():
        print("the model predicted", label_names[predicted_label], count, "times.")

    
    
    
    
    
    
    
    
    
    
    
    