# pip install transformers bitsandbytes pandas accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  

import pandas as pd
from huggingface_hub import login
hf_token=""
login(token=hf_token)


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
	model_name, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)


tokenizer = AutoTokenizer.from_pretrained(model_name)



# Load your test data and randomly select 10 samples
full_test_df = pd.read_csv('/workspace/clef2022-checkthat-task3/data/unpreprocessed/test.csv')
test_df = full_test_df.sample(n=10, random_state=42)  # Set random_state for reproducibility



with open("/workspace/clef2022-checkthat-task3/data/few_shot_prompt.txt", "r") as file:
    few_shot_examples = file.read()

# Function to generate prediction for a test sample
def classify_with_few_shot(test_text):
    # Create a prompt with few-shot examples and the test instance
    prompt = (few_shot_examples + f"\n\ntext: {test_text}\nlabel:   ")
    
    # Tokenize input prompt
    input_ids = tokenizer(prompt, return_tensors='pt').to("cuda")
    
    # Generate prediction from the model (autoregressive completion)
    output = quantized_model.generate(**input_ids, max_new_tokens=1)
    
    # Decode model output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("decoded_output:", decoded_output)
    
    # Extract the predicted label from the output
    predicted_label = last_word = decoded_output.split()[-1]
    print("predicted_label:", predicted_label)
    
    return predicted_label

# Iterate over test samples and classify
results = []
for _, row in test_df.iterrows():
    test_text = row['title_text']
    predicted_label = classify_with_few_shot(test_text)
    
    # Store results (test sample id and predicted label)
    results.append({'public_id': row['public_id'], 'predicted_label': predicted_label, 'label': row['label']})

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('./test_predictions.csv', index=False)
