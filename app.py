import os
import json
import pickle
import html

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from nlp_utils import analyze_text, ngram_counts, ensure_nltk


# ---------- Page ----------
st.set_page_config(
    page_title="Crowd Incident Analyzer",
    page_icon="🧠",
    layout="wide",
)

ensure_nltk()

ART = "artifacts"
DATA = "dataset.csv"


# ---------- Simple styling ----------
st.markdown(
    """
    <style>
    .title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        padding: 18px;
        background: rgba(128,128,128,.06);
    }
    .big {
        font-size: 1.35rem;
        font-weight: 750;
    }
    .muted {
        color: #6b7280;
        font-size: .9rem;
    }
    .pill {
        display: inline-block;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model ----------
@st.cache_resource
def load_assets():
    with open(os.path.join(ART, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(ART, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(ART, "similarity.pkl"), "rb") as f:
        similarity = pickle.load(f)
    model = load_model(os.path.join(ART, "crowd_lstm.keras"))
    return tokenizer, config, similarity, model


def model_ready():
    files = ["crowd_lstm.keras", "tokenizer.pkl", "config.json", "similarity.pkl"]
    return all(os.path.exists(os.path.join(ART, f)) for f in files)


def predict(text):
    tokenizer, config, _, model = load_assets()
    sequence = tokenizer.texts_to_sequences([text])
    x = pad_sequences(sequence, maxlen=config["max_len"], padding="post")
    probabilities = model.predict(x, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return (
        config["categories"][index],
        float(probabilities[index]),
        probabilities,
        config["categories"],
    )


def similar_incidents(text, k=5):
    _, _, similarity, _ = load_assets()
    query = similarity["vectorizer"].transform([text])
    scores = cosine_similarity(query, similarity["matrix"])[0]
    order = np.argsort(scores)[::-1][:k]
    return [
        (similarity["texts"][i], similarity["categories"][i], float(scores[i]))
        for i in order
    ]


def icon(category):
    return {
        "Stampede / Crushing Hazard": "🚨",
        "Violent Disorder / Brawl": "👊",
        "Medical Emergency / Overheating": "🏥",
        "Fire / Hazardous Smoke": "🔥",
        "Structural Failure / Collapse": "🏗️",
        "Normal Congestion / Safe Movement": "🚶",
    }.get(category, "📌")


def pills(items, limit=16):
    items = list(dict.fromkeys(items))[:limit]
    if not items:
        st.caption("No terms found.")
        return
    content = "".join(
        f'<span class="pill">{html.escape(str(item))}</span>' for item in items
    )
    st.markdown(content, unsafe_allow_html=True)


# ---------- Header ----------
st.markdown('<div class="title">🧠 Crowd Incident Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Analyze crowd incident text using Natural Language Processing</div>',
    unsafe_allow_html=True,
)


# ---------- Navigation ----------
with st.sidebar:
    st.markdown("## 🧠 Crowd Incident Analyzer")
    page = st.radio(
        "Menu",
        ["Dashboard", "Analyze Incident", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    if model_ready():
        st.success("Model ready")
    else:
        st.warning("Model not trained")


# ---------- Dashboard ----------
if page == "Dashboard":
    st.subheader("Welcome 👋")
    st.write(
        "Enter a crowd-related incident report and the system will classify it "
        "into one of six incident categories."
    )

    df = pd.read_csv(DATA)
    categories = list(df["category"].unique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Incident Records", f"{len(df):,}")
    c2.metric("Categories", len(categories))
    c3.metric("Model", "Ready" if model_ready() else "Not Ready")

    st.markdown("### Incident Categories")
    cols = st.columns(2)
    for i, category in enumerate(categories):
        with cols[i % 2]:
            st.markdown(
                f'<div class="card"><div class="big">{icon(category)} '
                f'{html.escape(category)}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### How it works")
    st.info("Incident text → NLP processing → LSTM classification → Result + text insights")

    if not model_ready():
        st.warning("Run `python train_model.py` before using the analyzer.")


# ---------- Analyze ----------
elif page == "Analyze Incident":
    st.subheader("Analyze an Incident")
    st.caption("Describe what happened in one or more sentences.")

    examples = {
        "Stampede": "Several people started pushing and running near the stadium exit while others were unable to move.",
        "Fire": "Thick smoke and flames were reported near a food stall while people moved away from the area.",
        "Medical": "A spectator collapsed after severe heat exhaustion and required medical assistance.",
        "Violence": "A fight broke out between two groups near the entrance and several people were pushing each other.",
        "Structural": "A metal barricade collapsed near the crowd and several people fell into the restricted area.",
        "Congestion": "A large number of visitors gathered near the entrance, causing congestion but movement remained controlled.",
    }

    st.markdown("**Try an example**")
    cols = st.columns(3)
    for i, (name, example) in enumerate(examples.items()):
        if cols[i % 3].button(name, use_container_width=True):
            st.session_state["incident_text"] = example

    text = st.text_area(
        "Incident description",
        key="incident_text",
        height=130,
        placeholder="Example: People are running and pushing near the exit.",
    )

    if st.button("🔍 Analyze Incident", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Please enter an incident description.")
        elif not model_ready():
            st.error("Model files are missing. Run `python train_model.py` first.")
        else:
            with st.spinner("Analyzing..."):
                info = analyze_text(text)
                category, confidence, probabilities, categories = predict(text)
                similarities = similar_incidents(text)

            st.session_state["result"] = {
                "text": text,
                "info": info,
                "category": category,
                "confidence": confidence,
                "probabilities": probabilities,
                "categories": categories,
                "similarities": similarities,
            }

    result = st.session_state.get("result")

    if result:
        info = result["info"]
        category = result["category"]
        confidence = result["confidence"]

        st.divider()
        st.markdown("### Analysis Result")

        r1, r2, r3 = st.columns([2.2, 1, 1])
        with r1:
            st.markdown(
                f'<div class="card"><div class="muted">PREDICTED INCIDENT</div>'
                f'<div class="big">{icon(category)} {html.escape(category)}</div></div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.metric("Model Confidence", f"{confidence * 100:.1f}%")
        with r3:
            st.metric("Language", info["language"].upper())

        st.markdown("### Text Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sentences", len(info["sentences"]))
        m2.metric("Words", len(info["words"]))
        m3.metric("Stopwords", len(info["words"]) - len(info["clean_words"]))
        m4.metric("Stemmed Terms", len(info["stems"]))

        st.markdown("### Important Terms")
        pills(info["clean_words"])

        # Keep detailed NLP output available without making the main page crowded.
        with st.expander("View NLP Details"):
            tab1, tab2, tab3, tab4 = st.tabs(
                ["N-grams", "POS", "Entities", "Similar Incidents"]
            )

            with tab1:
                counts = ngram_counts(info["clean_words"])
                a, b, c = st.tabs(["Unigrams", "Bigrams", "Trigrams"])
                for tab, name in zip((a, b, c), ("Unigrams", "Bigrams", "Trigrams")):
                    with tab:
                        rows = [
                            {"Pattern": " ".join(k), "Frequency": v}
                            for k, v in counts[name].most_common(10)
                        ]
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        else:
                            st.caption("Not enough words.")

            with tab2:
                pos_df = pd.DataFrame(info["pos"], columns=["Word", "POS"])
                if not pos_df.empty:
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No POS information found.")

            with tab3:
                entity_df = pd.DataFrame(info["entities"])
                if not entity_df.empty:
                    st.dataframe(entity_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No named entities detected.")

            with tab4:
                rows = [
                    {
                        "Incident": incident,
                        "Category": cat,
                        "Similarity": f"{score * 100:.1f}%",
                    }
                    for incident, cat, score in result["similarities"]
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------- About ----------
else:
    st.subheader("About")
    st.write(
        "**Crowd Incident Text Categorization Using Natural Language Processing** "
        "is a text-classification application that analyzes crowd incident descriptions "
        "and predicts one of six predefined categories."
    )

    st.markdown("### Categories")
    descriptions = {
        "Stampede / Crushing Hazard": "Crowd surges, pushing, trampling, or dangerous crowd pressure.",
        "Violent Disorder / Brawl": "Fights, clashes, or violent crowd behaviour.",
        "Medical Emergency / Overheating": "Collapse, injury, heat exhaustion, or medical emergencies.",
        "Fire / Hazardous Smoke": "Fire, smoke, burning material, or hazardous smoke.",
        "Structural Failure / Collapse": "Failure of barricades, gates, platforms, or other structures.",
        "Normal Congestion / Safe Movement": "Crowding where movement remains controlled.",
    }

    for category, description in descriptions.items():
        st.markdown(f"**{icon(category)} {category}** — {description}")

    st.markdown("### NLP Features")
    st.write(
        "Tokenization • Stopword filtering • Stemming • N-gram analysis • "
        "POS tagging • Named Entity Recognition • Similarity search • LSTM classification"
    )

    st.caption(
        "This is a text-based NLP prototype. It does not analyze CCTV, video, audio, or sensor data."
    )
