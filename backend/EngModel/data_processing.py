from datasets import load_dataset
import numpy as np

# Load your custom CSVs
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

def preprocess_function(examples):
    # Tokenize text
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    # Create multi-label (one-hot) format if needed
    # If your CSV 'label' is an integer, convert to one-hot:
    labels = np.zeros((len(examples["label"]), 8))
    for i, label in enumerate(examples["label"]):
        labels[i, label] = 1.0
    tokenized["labels"] = labels
    return tokenized

# Map processing to dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["test"]