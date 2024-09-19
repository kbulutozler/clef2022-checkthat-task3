import torch
from datasets import Dataset
import numpy as np
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import csv

import pandas as pd
from huggingface_hub import login

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support


### Data Prep
df = pd.read_csv('/root/clef2022-checkthat-task3/data/unpreprocessed/train.csv')
df_test = pd.read_csv('/root/clef2022-checkthat-task3/data/unpreprocessed/test.csv')

# Create id2label and label2id dictionaries
unique_labels = df['label'].unique()
label2id = {label: id for id, label in enumerate(unique_labels)}
id2label = {id: label for label, id in label2id.items()}

# Update df and df_test to use label IDs
df['label'] = df['label'].map(label2id)
df_test['label'] = df_test['label'].map(label2id)

df.columns = ['text', 'label']
df_test.columns = ['public_id','text', 'label']

#df_test = df_test.iloc[:10]


df = df.sample(frac=1).reset_index(drop=True)
df_train = df.iloc[:1000]
df_dev = df.iloc[1000:]

print(df_train.head())
print(df_dev.head())
print("\nTrain data label distribution:")
print(df_train['label'].value_counts(normalize=True))

print("\nDev data label distribution:")
print(df_dev['label'].value_counts(normalize=True))

train_dataset = Dataset.from_pandas(df_train)
dev_dataset = Dataset.from_pandas(df_dev)

print("\nTrain dataset:")
print(train_dataset)
print("\nDev dataset:")
print(dev_dataset)

### Model Configuration

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True, 
    bnb_4bit_compute_dtype = torch.bfloat16 
)

login(token="")

model_name = "meta-llama/Meta-Llama-3.1-8B"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=4,
    device_map='auto'
)

lora_config = LoraConfig(
    r = 16, 
    lora_alpha = 8,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, 
    bias = 'none',
    task_type = 'SEQ_CLS'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

### Helper Functions

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return calculate_metrics(labels, predictions)


def write_scores(test_df, csv_file):
    y_test = test_df.label
    y_pred = test_df.predictions
    
    metrics = calculate_metrics(y_test, y_pred)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Write results to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for metric, value in metrics.items():
            writer.writerow([metric.capitalize(), f"{value:.4f}"])


### Evaluate without training

sentences = df_test.text.tolist()

batch_size = 32  

all_outputs = []

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    print(i)
    inputs = tokenizer(batch_sentences, return_tensors="pt", 
    padding=True, truncation=True, max_length=768)

    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        all_outputs.append(outputs['logits'])
        
final_outputs = torch.cat(all_outputs, dim=0)
df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()


write_scores(df_test, 'wo_training.csv') # write results to CSV



### Training

def data_preprocesing(row):
    return tokenizer(row['text'], truncation=True, max_length=768)

tokenized_train = train_dataset.map(data_preprocesing, batched=True, remove_columns=['text'])
tokenized_train.set_format("torch")

tokenized_dev = dev_dataset.map(data_preprocesing, batched=True, remove_columns=['text'])
tokenized_dev.set_format("torch")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir = 'results',
    learning_rate = 1e-4,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 1,
    logging_steps=1,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

train_result = trainer.train()

### Evaluation after training

def generate_predictions(model,df_test):
    sentences = df_test.text.tolist()
    batch_size = 32  
    all_outputs = []

    for i in range(0, len(sentences), batch_size):

        batch_sentences = sentences[i:i + batch_size]

        inputs = tokenizer(batch_sentences, return_tensors="pt", 
        padding=True, truncation=True, max_length=768)

        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') 
        for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
        
    final_outputs = torch.cat(all_outputs, dim=0)
    df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()

generate_predictions(model,df_test)
write_scores(df_test, 'w_training.csv')




