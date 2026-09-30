import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

DATA = "dataset.csv"
ART = "artifacts"
os.makedirs(ART, exist_ok=True)

df = pd.read_csv(DATA)
categories = sorted(df["category"].unique())
label_to_id = {c:i for i,c in enumerate(categories)}
df["label"] = df["category"].map(label_to_id)

x_train, x_test, y_train, y_test = train_test_split(
    df["incident"].astype(str), df["label"],
    test_size=0.20, random_state=42, stratify=df["label"]
)

# Exp 9: TF-IDF for similarity
similarity_vectorizer = TfidfVectorizer(stop_words="english")
similarity_matrix = similarity_vectorizer.fit_transform(df["incident"])
with open(os.path.join(ART, "similarity.pkl"), "wb") as f:
    pickle.dump({
        "vectorizer": similarity_vectorizer,
        "matrix": similarity_matrix,
        "texts": df["incident"].tolist(),
        "categories": df["category"].tolist()
    }, f)

# Exp 10: Keras Tokenizer + padding + Embedding + LSTM
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train.tolist())
train_seq = tokenizer.texts_to_sequences(x_train.tolist())
test_seq = tokenizer.texts_to_sequences(x_test.tolist())
max_len = max(len(s) for s in train_seq)

X_train = pad_sequences(train_seq, maxlen=max_len, padding="post")
X_test = pad_sequences(test_seq, maxlen=max_len, padding="post")
y_train_np = np.asarray(y_train)
y_test_np = np.asarray(y_test)

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32),
    LSTM(32),
    Dense(len(categories), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train_np, epochs=20, verbose=0, validation_split=0.1)
loss, acc = model.evaluate(X_test, y_test_np, verbose=0)
pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

model.save(os.path.join(ART, "crowd_lstm.keras"))
with open(os.path.join(ART, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
with open(os.path.join(ART, "config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "categories": categories,
        "label_to_id": label_to_id,
        "max_len": int(max_len),
        "test_accuracy": float(acc)
    }, f, indent=2)

print("Training complete.")
print(f"Training records: {len(x_train)}")
print(f"Testing records: {len(x_test)}")
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test_np, pred, target_names=categories))
