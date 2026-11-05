'''
Naive Bayes email classifier.
Builds its own train/test split directly from the emails/ folder.
The trained_data.json file (produced by preprocessor.py) is intentionally
NOT used here — loading pre-computed counts from all emails would leak
test data into the model.
'''
import math
import os
import random
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.metrics import classification_report

# --- NLP Helpers ---

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


def load_junk_words(filepath):
    """Reads the stopword file and returns a set of words."""
    junk = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            junk = {line.strip().lower() for line in f if line.strip()}
    return junk


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


# --- Train/Test Split ---

def build_split(email_dir, test_fraction=0.2):
    """
    Reads the emails/ folder directly and returns a stratified 80/20 split.
    Returns:
        train_files: {category: [filepath, ...]}
        test_files:  {category: [filepath, ...]}
    """
    train_files = {}
    test_files = {}

    for category in os.listdir(email_dir):
        cat_path = os.path.join(email_dir, category)
        if not os.path.isdir(cat_path):
            continue

        files = [
            os.path.join(cat_path, f)
            for f in os.listdir(cat_path)
            if f.endswith('.txt')
        ]
        random.shuffle(files)

        n_test = math.ceil(len(files) * test_fraction)
        test_files[category]  = files[:n_test]
        train_files[category] = files[n_test:]

    return train_files, test_files


# --- Build Model from Training Files Only ---

def build_trained_data(train_files, lemmatizer, junk):
    """
    Computes word frequencies and email counts from the training split only.
    Produces the same structure as trained_data.json, but in memory,
    built exclusively from training emails so test emails stay unseen.
    """
    trained_data = {
        "metadata": {
            "category_email_counts": {},
            "total_emails": 0
        },
        "word_counts": {}
    }

    for category, files in train_files.items():
        trained_data["metadata"]["category_email_counts"][category] = len(files)
        trained_data["metadata"]["total_emails"] += len(files)

        cnt = Counter()
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            cnt.update(lemma(text, lemmatizer, junk))

        trained_data["word_counts"][category] = cnt

    return trained_data


# --- Core Classifier ---

def classify_email(processed_words, trained_data):
    """
    Runs Naive Bayes on a list of pre-processed words.
    Returns a dict of {category: log_probability_score}.
    """
    categories = trained_data["word_counts"].keys()
    email_counts = trained_data["metadata"]["category_email_counts"]
    total_emails = trained_data["metadata"]["total_emails"]

    all_unique_words = set()
    for cat in categories:
        all_unique_words.update(trained_data["word_counts"][cat].keys())
    vocab_size = len(all_unique_words)

    results = {}
    for cat in categories:
        prior = email_counts[cat] / total_emails
        score = math.log(prior)

        cat_word_data = trained_data["word_counts"][cat]
        total_word_count_in_cat = sum(cat_word_data.values())

        for word in processed_words:
            word_count = cat_word_data.get(word, 0)
            # Laplace smoothing
            word_probability = (word_count + 1) / (total_word_count_in_cat + vocab_size)
            score += math.log(word_probability)

        results[cat] = score

    return results


def calculate_confidence(scores):
    """
    Converts log-scores into normalized percentages (Softmax).
    """
    max_score = max(scores.values())
    exp_scores = {cat: math.exp(score - max_score) for cat, score in scores.items()}
    total_sum = sum(exp_scores.values())
    return {cat: (val / total_sum) * 100 for cat, val in exp_scores.items()}


# --- Public API ---

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_JUNK_PATH = os.path.join(_THIS_DIR, "junk.txt")
_EMAIL_DIR = os.path.join(_THIS_DIR, "emails")

_trained_data = None
_junk = None
_wnl = None


def _init():
    """Build model from training split once and cache it."""
    global _trained_data, _junk, _wnl
    if _trained_data is not None:
        return
    _junk = load_junk_words(_JUNK_PATH)
    _wnl = WordNetLemmatizer()
    train_files, _ = build_split(_EMAIL_DIR)
    _trained_data = build_trained_data(train_files, _wnl, _junk)


def classify_text(text: str):
    """
    Classify a raw email string.
    Returns winner and a dict of category confidences.
    """
    _init()
    processed_words = lemma(text, _wnl, _junk)
    scores = classify_email(processed_words, _trained_data)
    confidences = calculate_confidence(scores)
    winner = max(confidences, key=confidences.get)
    return winner, confidences


# --- Evaluation & Reporting ---

def run_performance_report():
    """
    Builds a fresh 80/20 split, trains on the 80%, and evaluates on the 20%.
    The model used here is independent of the cached _trained_data so the
    split is consistent within one report run.
    """
    junk = load_junk_words(_JUNK_PATH)
    wnl = WordNetLemmatizer()

    train_files, test_files = build_split(_EMAIL_DIR)
    trained_data = build_trained_data(train_files, wnl, junk)

    total_emails = sum(len(f) for f in train_files.values()) + sum(len(f) for f in test_files.values())
    total_train  = sum(len(f) for f in train_files.values())
    total_test   = sum(len(f) for f in test_files.values())

    print(f"Total Emails: {total_emails}")
    print(f"Training set: {total_train} | Test set: {total_test}")

    y_true = []
    y_pred = []

    for category, files in test_files.items():
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            processed_words = lemma(content, wnl, junk)
            scores = classify_email(processed_words, trained_data)
            confidences = calculate_confidence(scores)
            prediction = max(confidences, key=confidences.get)

            y_true.append(category)
            y_pred.append(prediction)

    print("\n--- Model Performance Report ---")
    print(classification_report(y_true, y_pred, zero_division=0))


# --- Script Mode ---

if __name__ == "__main__":
    new_email_path = os.path.join(_THIS_DIR, "new_email.txt")

    if os.path.exists(new_email_path):
        with open(new_email_path, 'r', encoding='utf-8') as f:
            content = f.read()

        winner, confidences = classify_text(content)
        confidence_val = confidences[winner]
        

    run_performance_report()
    print(f"\nFinal Result: {winner.upper()} ({confidence_val:.2f}% confidence)")