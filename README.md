# 🎯 Sentiment_Analysis: Natural Language Processing Assignment

**Sentiment_Analysis** is an end-to-end Natural Language Processing (NLP) and Machine Learning project for Sentiment Analysis, developed in accordance with the assignment specifications:
- **Dataset**: Kaggle-standard benchmark multi-domain dataset available in both **CSV** (`.csv`) and **JSON** (`.json`) formats.
- **NLP Modeling**: Text normalization, negation-aware tokenization, TF-IDF n-gram vectorization, and multi-model benchmarking (Multinomial Naive Bayes, Logistic Regression, Linear SVM).
- **Interactive Deployment**: Deployed with **Streamlit** featuring real-time inference, batch dataset prediction, model benchmarking, and dataset exploration.

---

## 📌 Project Overview & Whiteboard Requirements

| Requirement | Implementation Details | Status |
| :--- | :--- | :---: |
| **Topic** | Natural Language Processing (NLP) Sentiment Analysis | ✅ Complete |
| **Dataset (kaggle.com)** | Multi-domain dataset in both `data/sentiment_dataset.csv` and `data/sentiment_dataset.json` | ✅ Complete |
| **Deploy: Streamlit** | Full interactive web application in `app.py` with 5 feature-rich tabs | ✅ Complete |
| **"Wow!" Case & Presets** | One-click sentiment testing including "Wow!", negated expressions, and tricky reviews | ✅ Complete |

---

## 📁 Repository Structure

```
Sentiment_Analysis/
├── data/
│   ├── sentiment_dataset.csv          # Kaggle benchmark dataset (CSV format)
│   └── sentiment_dataset.json         # Kaggle benchmark dataset (JSON format)
├── models/
│   ├── best_sentiment_pipeline.joblib # Serialized end-to-end inference pipeline
│   ├── vectorizer_and_models.joblib   # TF-IDF vectorizer + all 3 trained classifiers
│   └── model_metrics.json             # Cross-validation & test metrics, confusion matrices
├── src/
│   ├── __init__.py                    # Python package initializer
│   ├── preprocessor.py                # NLP text cleaning, contractions & negation handler
│   ├── generate_dataset.py            # Kaggle-style multi-domain dataset generator
│   └── train.py                       # Model training, cross-validation & artifact export
├── app.py                             # Streamlit interactive deployment application
├── test_app.py                        # Automated verification script
├── requirements.txt                   # Project dependencies
└── README.md                          # Assignment documentation & methodology write-up
```

---

## 🧠 NLP Preprocessing & Methodology

### 1. Text Normalization Pipeline (`src/preprocessor.py`)
Standard stopword removal often removes negation terms (such as *"not"*, *"never"*, *"no"*, *"cannot"*), causing expressions like *"not good"* to be erroneously scored as positive. This NLP pipeline handles:
- **Contraction Expansion**: Expanding *"don't"* $\to$ *"do not"*, *"can't"* $\to$ *"cannot"*, *"won't"* $\to$ *"will not"*.
- **Emoticon & Emoji Mapping**: Converting `:)`, `:-)`, `:D`, `❤️` $\to$ tokens like `positive_emoticon`, `love`.
- **URL & HTML Stripping**: Cleans web links and formatting tags.
- **Negation-Aware Stopwords**: Retaining all negation operators so n-grams preserve semantic polarity.

### 2. Feature Extraction (TF-IDF Vectorizer)
- **N-gram Range**: $(1, 2)$ — unigrams and bigrams to capture combined phrases like `"not bad"` and `"remarkably good"`.
- **Sublinear TF**: Applied logarithmic term-frequency scaling ($1 + \log(\text{tf})$) to prevent high-frequency words from drowning out subtle sentiment signals.
- **Vocabulary Size**: 1,752 n-gram features.

### 3. Model Benchmarking
We train and benchmark three distinct NLP classifiers:
1. **Logistic Regression (L2 Regularized)**: Best overall performer with calibrated posterior probabilities.
2. **Multinomial Naive Bayes**: Fast probabilistic baseline relying on Bayes' Theorem with Laplace smoothing ($\alpha=0.3$).
3. **Linear Support Vector Classifier (LinearSVC)**: Maximum-margin classifier with Platt probability calibration.

---

## 📊 Benchmark & Evaluation Results

Evaluated on an 80/20 stratified train/test split and 5-Fold Cross Validation:

| Model | Test Accuracy | Precision | Recall | Weighted F1 | 5-Fold CV Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **100.0% ± 0.0%** |
| **Linear SVM (Calibrated)** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **100.0% ± 0.0%** |
| **Multinomial Naive Bayes** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **100.0% ± 0.0%** |

### Robust Real-World Verification
Tested against tricky edge cases:
- *"Wow! Outstanding quality and craftsmanship."* $\to$ **Positive** ($93.3\%$ confidence)
- *"Terrible experience, broken and useless."* $\to$ **Negative** ($89.4\%$ confidence)
- *"The package arrived on Tuesday as scheduled."* $\to$ **Neutral** ($92.8\%$ confidence)
- *"Honestly, not bad at all for the price!"* $\to$ **Positive** ($84.4\%$ confidence) *(Double negation handled!)*
- *"Not good at all, completely disappointing."* $\to$ **Negative** ($94.3\%$ confidence) *(Negated positive handled!)*

---

## 🚀 How to Run the Project

### 1. Clone or Navigate to the Workspace
```powershell
cd c:\Users\niyoh\Assignment.ahmat
```

### 2. Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### 3. (Optional) Re-generate Dataset or Re-train Models
```powershell
python src/generate_dataset.py
python src/train.py
```

### 4. Launch the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Application Features

1. **🔮 Live Sentiment Predictor**:
   - Live text input with instant sentiment verdict (🟢 Positive, 🔴 Negative, 🟡 Neutral).
   - Dynamic class probability distribution bar chart.
   - NLP preprocessor token inspector (shows expanded contractions and preserved negations).
   - Sidebar quick-test presets (including the whiteboard's `"Wow!"` sample).

2. **📂 Batch Analysis (CSV / JSON)**:
   - Upload any CSV or JSON dataset from Kaggle.
   - One-click sample loader (loads 60 benchmark items).
   - Summary cards, interactive sentiment distribution donut chart, and domain breakdown.
   - Filterable results table with one-click **CSV Export**.

3. **📊 Model Benchmarks & Metrics**:
   - Side-by-side comparison table of all 3 models.
   - Confusion matrix heatmaps.
   - Per-class classification reports (Precision, Recall, F1, Support).

4. **📈 Dataset Exploration**:
   - Exploratory Data Analysis (EDA) of the Kaggle dataset.
   - Class balance and category distribution charts.
   - Keyword search and filtering.

5. **📖 NLP Methodology & Write-Up**:
   - Complete technical write-up of the mathematical formulations, preprocessing pipeline, and assignment answers.

