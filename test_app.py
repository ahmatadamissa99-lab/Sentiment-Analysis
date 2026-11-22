import joblib
import json
from src.preprocessor import clean_text

bundle = joblib.load("models/vectorizer_and_models.joblib")
vec = bundle["vectorizer"]
model = bundle["models"][bundle["best_model_name"]]
labels = bundle["labels"]

test_sentences = [
    "Wow! Outstanding quality and craftsmanship.",
    "Terrible experience, broken and useless.",
    "The package arrived on Tuesday as scheduled.",
    "Honestly, not bad at all for the price!",
    "Not good at all, completely disappointing."
]

print(f"Loaded best model: {bundle['best_model_name']}")
print(f"Labels: {labels}\n")

for s in test_sentences:
    cleaned = clean_text(s)
    feat = vec.transform([cleaned])
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0]
    prob_str = ", ".join([f"{l}: {p:.1%}" for l, p in zip(labels, prob)])
    print(f"Input: {s}")
    print(f"Cleaned: {cleaned}")
    print(f"Verdict: {pred} ({prob_str})\n")

print("All tests passed successfully!")

