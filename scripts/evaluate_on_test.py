from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import pandas as pd
import torch
import os
from torchmetrics import F1Score
import csv

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
    results["f1_micro"] = f1_micro(predictions, labels).cpu().numpy().item()
    results["f1_macro"] = f1_macro(predictions, labels).cpu().numpy().item()
    return results

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # set it to cpu during debugging
batch_size=1
base_model_name = "distilbert-base-uncased" # or distilbert-base-uncased
model_path = os.path.join("/app/output", base_model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model_name)
print("tokenizer loaded")
dataset_path = "/app/data/preprocessed"
output_path = "/app/submissions"
label_names = {0:"false", 1:"partially false", 2: "true", 3:"other"}


model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)

model.to(device)
test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))

test_df = clean_dataset(test_df)
count = len(test_df)
test_results = {'public_id': [], 'predicted_rating': [], 'gold_label': []}
for index, row in test_df.iterrows():
    if index%15 == 0:
        print(f"{index}/{count}")
    tokenized_text = tokenizer(row['text'], padding="max_length", truncation=True, return_tensors="pt")
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
    with torch.no_grad():
        outputs = model(**tokenized_text)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).cpu().numpy().item()
    public_id = row['public_id']
    gold_label = row['labels']
    test_results['public_id'].append(public_id)
    test_results['gold_label'].append(gold_label)
    test_results['predicted_rating'].append(prediction)

scores = compute_metrics(torch.tensor(test_results['predicted_rating']), torch.tensor(test_results['gold_label']))

print(scores)
print("scores have been calculated. now write to file.")

file_path = os.path.join(output_path, 'subtask3_english_CoulterOzler.tsv')
with open(file_path, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter=',')
    tsv_writer.writerow(['public_id', 'predicted_rating'])
    for i in range(count):
        public_id = test_results['public_id'][i]
        predicted_rating = label_names[test_results['predicted_rating'][i]]
        tsv_writer.writerow([public_id, predicted_rating])
    

print("submission file is ready at", file_path)
    
    
    
    
    
    
    
    
    
    
    