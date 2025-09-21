from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load trained model
model_path = "./models/saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    label = torch.argmax(probs).item()
    return "FAKE" if label == 1 else "REAL", probs.tolist()

if __name__ == "__main__":
    examples = [
        "Breaking: Scientists discover water on Mars!",
        "Click here to win a free iPhone by filling this form!",
    ]
    for text in examples:
        label, probs = predict(text)
        print(f"{text} -> {label} ({probs})")
