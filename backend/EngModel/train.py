import os
import torch
import logging
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    AutoTokenizer
)

load_dotenv()

# Setup Logging
logging.basicConfig(
    filename="training_log.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Starting training on device: {device}")
print(f"Using device: {device}")

labels = [
    "admiration","amusement","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","embarrassment","excitement","gratitude",
    "grief","love","nervousness","optimism","pride","realization","relief","remorse"
]

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}

model_id = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True 
)

dataset = load_dataset('csv', data_files={'train': 'train_data.csv', 'test': 'test_data.csv'})

def preprocess_function(data):
    tokenized = tokenizer(data["text"], padding="max_length", truncation=True, max_length=128)
    labels_matrix = np.array([data[label] for label in labels]).T.astype(np.float32)
    tokenized["labels"] = labels_matrix
    return tokenized

tokenized_datasets = dataset.map(preprocess_function, batched=True)

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    fp16=torch.cuda.is_available(),
)

trainer = MultiLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

logger.info("Training started...")
trainer.train()
logger.info("Training completed successfully.")

# Save final model with filtered state_dict
print("Saving model as final_model.pt...")
state_dict = model.state_dict()

# Filter keys to match standard AutoModelForSequenceClassification expectations
filtered_state_dict = {
    k: v for k, v in state_dict.items() 
    if not any(prefix in k for prefix in ["emotion_head", "polarity_head", "intensity_head"])
}

checkpoint_payload = {
    "model_state_dict": filtered_state_dict,
    "config": model.config,
    "emotion_labels": labels,
    "model_name": model_id
}

# Ensure directory exists
os.makedirs("./model", exist_ok=True)
torch.save(checkpoint_payload, "./model/final_model.pt")
logger.info("Model saved in ./model/final_model.pt")
print("Training complete. Model saved in ./model/final_model.pt")