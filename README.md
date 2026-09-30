# Crowd Incident Text Categorization Using Natural Language Processing

A single NLP mini-project integrating the uploaded Experiments 1–10 into a Streamlit application.

## Experiments used

1. Dataset preparation / TF-IDF / train-test split
2. Sentence tokenization, word tokenization, filtering, stopword removal, language detection
3. Word tokenization and stopword removal
4. Porter stemming
5. Unigram, bigram and trigram generation
6. POS tagging
7. RegexpParser chunking
8. Named Entity Recognition
9. TF-IDF + cosine similarity
10. Keras Tokenizer + padding + Embedding + LSTM classification

## Project files

- `app.py` — Streamlit frontend
- `train_model.py` — trains the Exp 10 LSTM and prepares Exp 9 similarity artifacts
- `nlp_utils.py` — preprocessing, stemming, n-grams, POS, chunking and NER
- `dataset.csv` — 360-record demonstration dataset using the six categories in the project report
- `requirements.txt` — Python packages
- `artifacts/` — generated model/vectorizer files after training

## Important dataset note

The separate original 360-record dataset was not included among the uploaded experiment PDFs.
Therefore `dataset.csv` in this package is a **generated demonstration dataset** constructed
from the six project categories described in the existing mini-project report. It is not claimed
to be the original source dataset.

## Run

Use Python 3.10–3.12 for the easiest TensorFlow setup.

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the Streamlit address shown in the terminal.

## Expected workflow

Enter an incident such as:

> People are running and pushing near the exit.

The application then shows:

- predicted incident category
- confidence
- sentence/token information
- stopword count
- stemmed terms
- unigrams, bigrams and trigrams
- POS tags
- chunks
- named entities
- similar incidents using cosine similarity
