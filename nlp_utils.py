import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag, ne_chunk
from nltk.chunk import RegexpParser
from langdetect import detect, LangDetectException

def ensure_nltk():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        "chunkers/maxent_ne_chunker_tab": "maxent_ne_chunker_tab",
        "corpora/words": "words",
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

def analyze_text(text):
    ensure_nltk()
    text = text.strip()
    sentences = sent_tokenize(text) if text else []
    words = word_tokenize(text) if text else []
    filtered = [w for w in words if w.isalpha()]
    stop_words = set(stopwords.words("english"))
    clean_words = [w for w in filtered if w.lower() not in stop_words]

    stemmer = PorterStemmer()
    stems = [stemmer.stem(w) for w in clean_words]

    try:
        language = detect(text) if text else "unknown"
    except LangDetectException:
        language = "unknown"

    try:
        pos = pos_tag(words)
    except Exception:
        pos = []

    try:
        entities = ne_chunk(pos)
    except Exception:
        entities = None

    grammar = r"""
        NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
        VP: {<VB.*><RB.*>*<VB.*>*}
    """
    chunks = []
    try:
        parser = RegexpParser(grammar)
        tree = parser.parse(pos)
        for subtree in tree.subtrees():
            if subtree.label() in ("NP", "VP"):
                chunks.append({
                    "type": subtree.label(),
                    "text": " ".join(word for word, _ in subtree.leaves())
                })
    except Exception:
        pass

    named_entities = []
    if entities is not None:
        for subtree in entities:
            if hasattr(subtree, "label"):
                named_entities.append({
                    "text": " ".join(word for word, _ in subtree.leaves()),
                    "label": subtree.label()
                })

    return {
        "sentences": sentences,
        "words": words,
        "filtered_words": filtered,
        "clean_words": clean_words,
        "stems": stems,
        "language": language,
        "pos": pos,
        "entities": named_entities,
        "chunks": chunks,
    }

def make_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))]

def ngram_counts(words):
    from collections import Counter
    result = {}
    for n, name in [(1, "Unigrams"), (2, "Bigrams"), (3, "Trigrams")]:
        result[name] = Counter(make_ngrams(words, n))
    return result
