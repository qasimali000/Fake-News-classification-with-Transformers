import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import inspect

# ✅ Use our custom loader
from data_loader import load_and_prepare_data

# ------------------------------
# Data loading
# ------------------------------
train_texts, test_texts, train_labels, test_labels = load_and_prepare_data()

train_df = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_df = Dataset.from_dict({"text": test_texts, "label": test_labels})

# ------------------------------
# Tokenizer & preprocessing
# ------------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_dataset = train_df.map(tokenize, batched=True)
test_dataset = test_df.map(tokenize, batched=True)

# Hugging Face Trainer expects "labels" column
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ------------------------------
# Model
# ------------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ------------------------------
# Metrics
# ------------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"],
    }

# ------------------------------
# Compatibility wrapper for TrainingArguments
# ------------------------------
def make_training_args():
    params = inspect.signature(TrainingArguments.__init__).parameters

    kwargs = {
        "output_dir": "./models/saved_model",
        "logging_dir": "./logs",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 2,  # change to 3–5 for better accuracy
        "weight_decay": 0.01,
        "logging_steps": 20,
    }

    # evaluation strategy param name may differ
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"

    # saving strategy
    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"

    # load best model
    if "load_best_model_at_end" in params:
        kwargs["load_best_model_at_end"] = True

    return TrainingArguments(**kwargs)

training_args = make_training_args()

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ------------------------------
# Train & Save
# ------------------------------
trainer.train()
trainer.save_model("./models/saved_model")
tokenizer.save_pretrained("./models/saved_model")
