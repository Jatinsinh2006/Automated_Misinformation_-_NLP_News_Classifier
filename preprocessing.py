"""
preprocessing.py
-----------------
Text cleaning + preprocessing utilities for the Fake News Detection project.

Steps applied:
1. Lowercase
2. Remove URLs, HTML tags, punctuation, numbers
3. Tokenization
4. Stopword removal
5. Stemming (Porter Stemmer)
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only runs once, cached after that)
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    """Lowercase + strip URLs/HTML/punctuation/numbers."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                      # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                    # numbers + punctuation
    text = re.sub(r"\s+", " ", text).strip()                 # extra whitespace
    return text


def tokenize_and_stem(text: str) -> str:
    """Tokenize, remove stopwords, apply stemming, and rejoin into a string."""
    tokens = word_tokenize(text)
    tokens = [
        STEMMER.stem(tok) for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    """Full pipeline: clean -> tokenize -> stopword removal -> stemming."""
    cleaned = clean_text(text)
    return tokenize_and_stem(cleaned)


def preprocess_dataframe(df, text_column="text", new_column="clean_text"):
    """Apply preprocess_text() to an entire dataframe column, with a progress bar."""
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Preprocessing")
        df[new_column] = df[text_column].progress_apply(preprocess_text)
    except ImportError:
        # Fallback if tqdm isn't installed
        df[new_column] = df[text_column].apply(preprocess_text)
    return df


if __name__ == "__main__":
    sample = "BREAKING!!! Scientists in the U.S. discover 100% proof that <b>aliens</b> exist! Visit http://fake-news.com now!!"
    print("Original :", sample)
    print("Cleaned  :", preprocess_text(sample))