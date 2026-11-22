"""
NLP Text Preprocessing Module for Sentiment Analysis
Handles contractions, cleaning, URL/tag removal, punctuation, and negation-aware tokenization.
"""

import re
import string

# Contraction mapping
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "cannot": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
}

# Emoticon mapping
EMOTICON_MAP = {
    r":\)|:-\)|:\*\)|:\]|;-\)|;\)": " positive_emoticon ",
    r":D|:-D|8D|8-D|xD|XD": " positive_emoticon ",
    r":\(|:-\(|:\[|;-\(|;\(": " negative_emoticon ",
    r":'\(|:'-\(": " negative_emoticon ",
    r":/|:-/|:\\|:-\\": " neutral_emoticon ",
    r"<3": " love ",
    r"❤️|💖|💕": " love ",
    r"👍": " thumbs_up ",
    r"👎": " thumbs_down ",
    r"🔥": " fire ",
    r"⭐": " star ",
}

# Negation words to preserve during stopword filtering
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "hardly", "barely",
    "scarcely", "cannot", "without", "against", "lack", "lacks", "lacking"
}

# Base stopwords list (customized to retain negations)
BASE_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

# Stopwords after excluding negations
SENTIMENT_STOPWORDS = list(BASE_STOPWORDS - NEGATION_WORDS)


def expand_contractions(text: str) -> str:
    """Expand standard English contractions."""
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b", re.IGNORECASE)
    
    def replace(match):
        return CONTRACTION_MAP.get(match.group(0).lower(), match.group(0))
        
    for k, v in CONTRACTION_MAP.items():
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Full NLP cleaning pipeline for raw text:
    1. Type safety check
    2. HTML tags removal
    3. URLs and user mentions removal
    4. Emoticon & emoji expansion
    5. Contractions expansion
    6. Repeated characters reduction (e.g. 'loooove' -> 'love')
    7. Punctuation removal / normalization
    8. Whitespace trimming & lowercase
    9. Optional stopword removal (preserving negations)
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove user mentions and email addresses
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Convert emoticons
    for pattern, replacement in EMOTICON_MAP.items():
        text = re.sub(pattern, replacement, text)

    # Expand contractions
    text = expand_contractions(text)

    # Remove hashtags symbol while keeping text
    text = re.sub(r"#(\w+)", r"\1", text)

    # Reduce character repeats (e.g., 'soooo' -> 'soo', 'greeeeat' -> 'great')
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove numbers and punctuation, keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s_]", " ", text)

    # Strip and collapse multiple spaces
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in SENTIMENT_STOPWORDS and len(t) > 1]

    return " ".join(tokens)

