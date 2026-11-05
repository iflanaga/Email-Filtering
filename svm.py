import os
import math
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_files
from sklearn.metrics import classification_report

# --- NLP Helpers (consistent with nb.py) ---

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_JUNK_PATH = os.path.join(_THIS_DIR, "junk.txt")

def load_junk_words(filepath):
    """Reads the stopword file and returns a set of words."""
    junk = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            junk = {line.strip().lower() for line in f if line.strip()}
    return junk

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def lemma(text, lemmatizer, junk):
    """Tokenizes, POS-tags, and lemmatizes a raw string of text."""
    tokens = word_tokenize(text.lower())
    tagged_tokens = pos_tag(tokens)
    final_lemmas = []
    for word, tag in tagged_tokens:
        if word.isalnum():
            w_pos = get_wordnet_pos(tag)
            lem = lemmatizer.lemmatize(word, w_pos)
            if lem not in junk and len(lem) > 1:
                final_lemmas.append(lem)
    return final_lemmas

# Build shared NLP resources once
_junk = load_junk_words(_JUNK_PATH)
_wnl = WordNetLemmatizer()

def lemma_tokenizer(text):
    """
    Drop-in tokenizer for TfidfVectorizer.
    Returns a list of lemmatized, filtered tokens — identical pipeline to nb.py.
    """
    return lemma(text, _wnl, _junk)

# --- 1. Load data ---
data = load_files('emails', encoding='utf-8', decode_error='replace')
X = np.array(data.data)
y = np.array(data.target)
target_names = data.target_names

X_train_list, X_test_list = [], []
y_train_list, y_test_list = [], []

# --- 2. Manual Stratified Split per Category ---
unique_classes = np.unique(y)

for cls in unique_classes:
    indices = np.where(y == cls)[0]
    np.random.shuffle(indices)

    total_in_cat = len(indices)
    n_test = math.ceil(total_in_cat * 0.2)
    n_train = total_in_cat - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train_list.extend(X[train_idx])
    X_test_list.extend(X[test_idx])
    y_train_list.extend(y[train_idx])
    y_test_list.extend(y[test_idx])

X_train, X_test = X_train_list, X_test_list
y_train, y_test = np.array(y_train_list), np.array(y_test_list)

print(f"Total Emails: {len(X)}")
print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")

# --- 3. Create and Train Pipeline ---
# tokenizer= replaces sklearn's default tokenizer with our lemmatization pipeline.
# stop_words is left as None because junk.txt filtering is already handled inside lemma_tokenizer.
model = make_pipeline(
    TfidfVectorizer(tokenizer=lemma_tokenizer, stop_words=None, ngram_range=(1, 2)),
    SVC(kernel='linear', probability=True)
)

model.fit(X_train, y_train)

# --- 4. Predict and Report ---
y_pred = model.predict(X_test)

print("\n--- Accuracy Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))


def classify_new_email(file_path):
    if not os.path.exists(file_path):
        return "File not found.", 0
    with open(file_path, 'r', encoding='utf-8') as f:
        content = [f.read()]
    prediction = model.predict(content)
    probs = model.predict_proba(content)
    return target_names[prediction[0]], max(probs[0]) * 100

# --- 5. Test the new email ---
res, conf = classify_new_email("new_email.txt")
print(f"\nFinal Result: {res.upper()} ({conf:.2f}% confidence)")