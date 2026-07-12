import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

# 1. Define your labels
labels = ["anger", "joy", "sadness", "fear", "surprise", "disgust", "neutral", "romantic"]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}

# 2. Load model with new head
model = AutoModelForSequenceClassification.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # Replaces the 7-label head with your new head
)

# 3. Setup Trainer (Assuming you have 'train_dataset' and 'val_dataset' ready)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 4. Train
trainer.train()

# 5. Export to .pt format
# The model weights are stored in model.state_dict()
torch.save(model.state_dict(), "sentiment_model.pt")
print("Model saved as sentiment_model.pt")