import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(fake_path="data/Fake.csv", true_path="data/True.csv"):
    # Load datasets
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # Add labels
    fake["label"] = 1
    true["label"] = 0

    # Combine
    df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Keep only text + label
    df = df[["text", "label"]]

    # Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
    )

    return train_texts, test_texts, train_labels, test_labels

if __name__ == "__main__":
    train_texts, test_texts, train_labels, test_labels = load_and_prepare_data()
    print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")
