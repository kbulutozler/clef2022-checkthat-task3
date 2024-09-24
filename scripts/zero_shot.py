import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, precision_recall_fscore_support
import csv
from huggingface_hub import login
login(token="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")  # Replace with your actual token

### Data Preparation

# Load your datasets
df = pd.read_csv('../data/unpreprocessed/train.csv')
df_test = pd.read_csv('../data/unpreprocessed/test.csv')
# Create label mappings
unique_labels = df['label'].unique()
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Map labels to IDs
df['label'] = df['label'].map(label2id)
df_test['label'] = df_test['label'].map(label2id)

# Rename columns for consistency
df.columns = ['text', 'label']
df_test.columns = ['public_id','text', 'label']

# Shuffle and split the data
df_train = df.sample(frac=1).reset_index(drop=True)



### Model and Tokenizer Setup

model_name = "meta-llama/Meta-Llama-3.1-8B"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure padding token is set
tokenizer.pad_token = tokenizer.eos_token

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the causal language model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    quantization_config=quantization_config,
    trust_remote_code=True
)

model.eval()  # Set model to evaluation mode

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

def generate_prompt(new_input):
    prompt = "Categorize this text into one of the 4 labels: truth, fake, partial, other. partial means partial truth/fake. Go!\n\n"
    prompt += f"Text: {new_input}\nLabel:"
    return prompt

def map_output_to_label(output_text, id2label):
    # Extract the label from the generated text
    # This assumes the model generates the label name directly after "Label:"
    label = output_text.strip().split('\n')[0]
    # Find the label that best matches the generated text
    for key in id2label.values():
        if key.lower() in label.lower():
            return key
    return None  # Or a default label

### Zero-Shot Learning Inference

def zero_shot_predict(model, tokenizer, text, temperature=0, top_p=0.95):
    prompt = generate_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,  # Adjust based on expected label length
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the label
    label = map_output_to_label(generated_text.split("Label:")[-1], id2label)
    return label

### Evaluation

def evaluate_zero_shot(model, tokenizer, df_test, id2label, label2id):
    y_true = []
    y_pred = []
    
    for idx, row in df_test.iterrows():
        text = row['text']
        true_label = id2label[row['label']]
        
        # Predict using zero-shot learning
        predicted_label = zero_shot_predict(model, tokenizer, text)
        
        if predicted_label is None:
            # Handle cases where the model's output doesn't map to any label
            predicted_label = "Unknown"
        
        y_true.append(row['label'])
        y_pred.append(label2id.get(predicted_label, -1))  # Assign -1 for unknown labels
    
    # Add predictions to the dataframe
    df_test = df_test.copy()
    df_test['predictions'] = y_pred
    
    # Remove instances with unknown predictions if necessary
    df_test = df_test[df_test['predictions'] != -1]
    
    # Write scores to CSV
    write_scores(df_test, 'zero_shot_results.csv')

### Main Execution

if __name__ == "__main__":
    # Evaluate on the test set using zero-shot learning
    evaluate_zero_shot(model, tokenizer, df_test, id2label, label2id)
