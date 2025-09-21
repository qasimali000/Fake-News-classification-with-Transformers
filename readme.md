# Fake News Detection with Transformers

Detect fake news using **DistilBERT**. This repository includes scripts to train, evaluate, and run inference on news articles labeled as **REAL** or **FAKE**.

---

## Setup

1. Clone the repo:

```bash
git clone https://github.com/qasimali000/Fake-News-classification-with-Transformers.git
cd Fake-News-classification-with-Transformers
```

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the **Fake & Real News Dataset** from Kaggle and place `Fake.csv` and `True.csv` in the `data/` folder:
   [Kaggle dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## Usage

* **Check data loader**

```bash
python data_loader.py
```

* **Train the model**

```bash
python fake_news_classifier.py
```

* **Run inference**

```bash
python inference.py
```

---

## Results

* Accuracy: \~92%
* F1 Score: \~91%

---

## Author

**Qasim Ali** — [Medium](https://medium.com/@capali) | [GitHub](https://github.com/qasimali000)
