"""
Sentiment Analysis Application
Natural Language Processing (NLP) Classification System
Trained on Kaggle Benchmark Multi-Domain Sentiment Dataset
Theme: Modern Dark Mode (Black, Electric Violet, Royal Blue)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import NLP Preprocessing module
try:
    from src.preprocessor import clean_text, expand_contractions, NEGATION_WORDS
except ImportError:
    from preprocessor import clean_text, expand_contractions, NEGATION_WORDS

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark UI: Black (#090A10), Electric Violet (#8B5CF6), and Royal Blue (#2563EB)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Dark Theme */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #090A10 !important;
        color: #F8FAFC !important;
    }

    /* Sidebar Dark Styling */
    section[data-testid="stSidebar"] {
        background-color: #0E1019 !important;
        border-right: 1px solid rgba(139, 92, 246, 0.18) !important;
    }
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #94A3B8 !important;
    }

    /* Header with Black, Violet & Blue Accents */
    .app-header {
        padding: 0.6rem 0 1.4rem 0;
        border-bottom: 1px solid rgba(139, 92, 246, 0.2);
        margin-bottom: 1.8rem;
        position: relative;
    }
    .app-header::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 140px;
        height: 2px;
        background: linear-gradient(90deg, #8B5CF6, #3B82F6);
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.6);
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF 30%, #C084FC 70%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        font-size: 0.96rem;
        color: #94A3B8 !important;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Modern Dark Cards */
    .card-panel {
        background: #111322;
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
        margin-bottom: 1rem;
    }

    /* Verdict Display Container */
    .verdict-container {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(139, 92, 246, 0.25);
        background: #111322;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }
    .verdict-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #8B5CF6 0%, #3B82F6 100%);
    }

    .badge-positive {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .badge-negative {
        display: inline-flex;
        align-items: center;
        background: rgba(244, 63, 94, 0.14);
        color: #FB7185;
        border: 1px solid rgba(251, 113, 133, 0.35);
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .badge-neutral {
        display: inline-flex;
        align-items: center;
        background: rgba(139, 92, 246, 0.14);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.35);
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .verdict-headline {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 0.75rem;
        margin-bottom: 0.2rem;
        color: #FFFFFF !important;
        letter-spacing: -0.01em;
    }
    .verdict-confidence {
        font-size: 0.92rem;
        color: #94A3B8 !important;
        margin-bottom: 1.2rem;
    }

    /* KPI Stat Boxes */
    .stat-box {
        background: #111322;
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }
    .stat-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #8B5CF6, #3B82F6);
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #A5B4FC !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-number {
        font-size: 1.75rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-top: 0.3rem;
    }
    .stat-note {
        font-size: 0.75rem;
        color: #64748B !important;
        margin-top: 0.2rem;
    }

    /* Token Chips in Dark Mode */
    .token-tag {
        display: inline-block;
        background: #181B2E;
        color: #C7D2FE;
        border: 1px solid rgba(139, 92, 246, 0.25);
        padding: 3px 9px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px 4px;
    }
    .token-negation {
        display: inline-block;
        background: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px 4px;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.2);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(139, 92, 246, 0.2) !important;
        padding-bottom: 2px;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 9px 18px;
        font-weight: 500;
        font-size: 0.92rem;
        color: #94A3B8 !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #C084FC !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #8B5CF6 !important;
        text-shadow: 0 0 10px rgba(139, 92, 246, 0.4);
    }

    /* Primary Buttons (Violet to Blue Gradient) */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7C3AED 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary Buttons */
    .stButton > button {
        background-color: #131522 !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        border-color: #8B5CF6 !important;
        color: #FFFFFF !important;
    }

    /* Form Inputs & Textarea */
    .stTextArea textarea {
        background-color: #111322 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(139, 92, 246, 0.28) !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.35) !important;
    }
    .stTextInput input {
        background-color: #111322 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(139, 92, 246, 0.28) !important;
        border-radius: 8px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #111322 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(139, 92, 246, 0.28) !important;
        border-radius: 8px !important;
    }

    /* Dataframe Table Dark Theme */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 8px !important;
        background-color: #0E1019 !important;
    }

    /* Headings */
    h3, h4, h5 {
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Resource Loaders (Cached)
# ==========================================
@st.cache_resource
def load_model_artifacts():
    models_bundle_path = os.path.join("models", "vectorizer_and_models.joblib")
    metrics_path = os.path.join("models", "model_metrics.json")
    
    if not os.path.exists(models_bundle_path):
        st.error("Model artifacts not found. Please run `python src/train.py`.")
        return None, None
        
    bundle = joblib.load(models_bundle_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    return bundle, metrics


@st.cache_data
def load_dataset():
    csv_path = os.path.join("data", "sentiment_dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


bundle, metrics_data = load_model_artifacts()
dataset_df = load_dataset()

# ==========================================
# Sidebar: Settings & Presets
# ==========================================
with st.sidebar:
    st.markdown("#### Classifier Settings")
    
    if bundle is not None:
        available_models = list(bundle["models"].keys())
        default_idx = available_models.index(bundle["best_model_name"]) if bundle["best_model_name"] in available_models else 0
        
        selected_model_name = st.selectbox(
            "Active Model",
            options=available_models,
            index=default_idx,
            help="Choose between supervised algorithms trained on TF-IDF n-grams."
        )

        st.caption(f"Vocabulary: **{metrics_data.get('vocab_size', 1752):,}** n-gram features")
        st.markdown("---")

        st.markdown("#### Preprocessing")
        remove_stopwords = st.checkbox(
            "Filter stopwords (preserve negations)",
            value=False,
            help="Removes common stopwords while strictly retaining 'not', 'never', 'cannot'."
        )
        show_token_debug = st.checkbox("Show preprocessing inspection", value=True)

        st.markdown("---")
        st.markdown("#### Test Presets")
        st.caption("Click to evaluate sample inputs:")

        preset_chosen = None
        if st.button("Positive: 'Wow! Fantastic movie & cast'"):
            preset_chosen = "Wow! Fantastic movie & cast. Truly outstanding performance and cinematography!"
        if st.button("Negative: 'Worst customer service ever'"):
            preset_chosen = "Worst customer service ever. Broken item arrived and nobody responds to emails. Horrible!"
        if st.button("Neutral: 'Standard packaging, average build'"):
            preset_chosen = "The package arrived on Tuesday as scheduled in standard brown cardboard. Works as expected."
        if st.button("Double Negation: 'Not bad at all for the price'"):
            preset_chosen = "Honestly, not bad at all for the price! I cannot complain, it exceeded expectations."
        if st.button("Negated Positive: 'Not impressed by battery life'"):
            preset_chosen = "I was not impressed by the battery life. Died within 3 hours. Will not recommend."

    st.markdown("---")
    st.caption("Sentiment Analysis • NLP Platform")


# ==========================================
# Application Header
# ==========================================
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Sentiment Analysis</h1>
    <p class="app-subtitle">
        Natural Language Processing system using text normalization, negation-aware TF-IDF vectorization,
        and multi-model supervised classification.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# Tabs Navigation
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Analysis",
    "Batch Processing",
    "Model Evaluation",
    "Dataset Explorer",
    "Documentation"
])


# ==========================================
# Inference Function
# ==========================================
def predict_sentiment(text, model_name, remove_stop=False):
    if bundle is None:
        return None
    
    vectorizer = bundle["vectorizer"]
    model = bundle["models"][model_name]
    labels = bundle["labels"]
    
    cleaned = clean_text(text, remove_stopwords=remove_stop)
    vec = vectorizer.transform([cleaned])
    
    pred_label = model.predict(vec)[0]
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(vec)[0]
        exp_d = np.exp(d - np.max(d))
        probs = exp_d / exp_d.sum()
    else:
        probs = [1.0 if l == pred_label else 0.0 for l in labels]
        
    prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    confidence = float(max(probs))
    
    words = cleaned.split()
    matched_ngrams = []
    vocab = vectorizer.vocabulary_
    for w in words:
        if w in vocab:
            matched_ngrams.append(w)
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in vocab:
            matched_ngrams.append(bigram)

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "cleaned_text": cleaned,
        "tokens": words,
        "matched_ngrams": matched_ngrams
    }


# =============================================================================
# TAB 1: LIVE ANALYSIS
# =============================================================================
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("##### Input Text")
        default_text = preset_chosen if preset_chosen else "Wow! This is remarkably amazing, exceeded all my expectations!"
        user_input = st.text_area(
            "Review or sentence to evaluate:",
            value=default_text,
            height=130,
            placeholder="Type any text or review here...",
            label_visibility="collapsed"
        )

        char_count = len(user_input)
        word_count = len(user_input.split())
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: -6px; margin-bottom: 12px;">
            {word_count} words &bull; {char_count} characters
        </div>
        """, unsafe_allow_html=True)

        analyze_btn = st.button("Analyze Sentiment", type="primary")

    with col_result:
        st.markdown("##### Prediction Result")
        if user_input.strip():
            result = predict_sentiment(user_input, selected_model_name, remove_stop=remove_stopwords)
            pred = result["prediction"]
            conf = result["confidence"]
            probs = result["probabilities"]

            badge_style = f"badge-{pred.lower()}"

            st.markdown(f"""
            <div class="verdict-container">
                <span class="{badge_style}">&bull; {pred} Sentiment</span>
                <div class="verdict-headline">{pred}</div>
                <div class="verdict-confidence">Model Confidence: <b style="color: #60A5FA;">{conf * 100:.1f}%</b></div>
            """, unsafe_allow_html=True)

            # Modern Violet & Blue horizontal probability progress bars
            for sentiment_name, prob_val in probs.items():
                pct = prob_val * 100
                if sentiment_name == "Positive":
                    bar_gradient = "linear-gradient(90deg, #059669, #10B981)"
                elif sentiment_name == "Negative":
                    bar_gradient = "linear-gradient(90deg, #DC2626, #F43F5E)"
                else:
                    bar_gradient = "linear-gradient(90deg, #7C3AED, #3B82F6)"
                    
                st.markdown(f"""
                <div style="margin-bottom: 9px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;">
                        <span style="font-weight: 500; color: #E2E8F0;">{sentiment_name}</span>
                        <span style="font-weight: 700; color: #FFFFFF;">{pct:.1f}%</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.08); border-radius: 4px; height: 7px; overflow: hidden;">
                        <div style="background: {bar_gradient}; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Preprocessing Breakdown
    if user_input.strip() and show_token_debug:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Preprocessing & Feature Extraction")
        
        col_t1, col_t2 = st.columns(2, gap="medium")
        with col_t1:
            st.markdown("<span style='font-size: 0.82rem; font-weight: 600; color: #A5B4FC;'>NORMALIZED TEXT</span>", unsafe_allow_html=True)
            st.code(result["cleaned_text"], language="text")

        with col_t2:
            st.markdown("<span style='font-size: 0.82rem; font-weight: 600; color: #A5B4FC;'>TOKENIZED UNITS</span>", unsafe_allow_html=True)
            if result["tokens"]:
                tokens_html = ""
                for t in result["tokens"]:
                    if t in NEGATION_WORDS:
                        tokens_html += f'<span class="token-negation">{t} [negation]</span>'
                    else:
                        tokens_html += f'<span class="token-tag">{t}</span>'
                st.markdown(tokens_html, unsafe_allow_html=True)
            else:
                st.caption("No tokens extracted.")


# =============================================================================
# TAB 2: BATCH PROCESSING
# =============================================================================
with tab2:
    st.markdown("##### Batch Dataset Evaluation")
    st.caption("Upload a CSV or JSON dataset from Kaggle, or evaluate with the pre-loaded 60-sample benchmark subset.")

    col_upload, col_sample = st.columns([3, 1.2], gap="medium")
    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"])
    with col_sample:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        use_sample_btn = st.button("Load Benchmark Sample (60 rows)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_data = json.load(uploaded_file)
                if isinstance(batch_data, list):
                    batch_df = pd.DataFrame(batch_data)
                elif isinstance(batch_data, dict):
                    batch_df = pd.DataFrame(batch_data)
                else:
                    batch_df = pd.DataFrame()
            st.success(f"Loaded {len(batch_df)} rows from `{uploaded_file.name}`.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif use_sample_btn or ("use_benchmark_sample" in st.session_state and st.session_state["use_benchmark_sample"]):
        st.session_state["use_benchmark_sample"] = True
        if dataset_df is not None:
            batch_df = dataset_df.sample(n=min(60, len(dataset_df)), random_state=42).copy()

    if batch_df is not None and not batch_df.empty:
        # Sanitize column names
        clean_cols = []
        for i, c in enumerate(batch_df.columns):
            c_str = str(c).strip() if pd.notna(c) else ""
            if not c_str or c_str.lower() in ("nan", "none") or c_str.startswith("Unnamed:"):
                clean_cols.append(f"Column_{i+1}")
            else:
                clean_cols.append(c_str)
        batch_df.columns = clean_cols

        # Identify candidate text columns
        text_cols = [c for c in batch_df.columns if batch_df[c].dtype == "object" or pd.api.types.is_string_dtype(batch_df[c])]
        if not text_cols:
            text_cols = list(batch_df.columns)

        default_col_idx = 0
        preferred = ["text", "review", "content", "comment", "tweet", "sentence", "body"]
        for p in preferred:
            matches = [i for i, c in enumerate(text_cols) if p in c.lower()]
            if matches:
                default_col_idx = matches[0]
                break

        col_c1, col_c2 = st.columns([3, 1], gap="medium")
        with col_c1:
            selected_text_col = st.selectbox(
                "Target Text Column",
                options=text_cols,
                index=default_col_idx if default_col_idx < len(text_cols) else 0
            )
        with col_c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_batch = st.button("Run Predictions", type="primary", use_container_width=True)

        if run_batch:
            with st.spinner("Classifying batch..."):
                raw_texts = batch_df[selected_text_col].fillna("").astype(str).tolist()
                cleaned = [clean_text(t, remove_stopwords=remove_stopwords) for t in raw_texts]
                vecs = bundle["vectorizer"].transform(cleaned)
                model = bundle["models"][selected_model_name]
                preds = model.predict(vecs)
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vecs)
                    confs = np.max(probs, axis=1)
                else:
                    confs = [1.0] * len(preds)

                batch_df["Predicted_Sentiment"] = preds
                batch_df["Confidence"] = [round(float(c) * 100, 1) for c in confs]
                st.session_state["batch_results"] = batch_df

        if "batch_results" in st.session_state:
            res_df = st.session_state["batch_results"]
            st.markdown("---")
            
            total_n = len(res_df)
            pos_cnt = (res_df["Predicted_Sentiment"] == "Positive").sum()
            neg_cnt = (res_df["Predicted_Sentiment"] == "Negative").sum()
            neu_cnt = (res_df["Predicted_Sentiment"] == "Neutral").sum()

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-label">Total Analyzed</div>
                    <div class="stat-number">{total_n:,}</div>
                    <div class="stat-note">100% processed</div>
                </div>
                """, unsafe_allow_html=True)
            with s2:
                st.markdown(f"""
                <div class="stat-box" style="border-color: rgba(16, 185, 129, 0.4);">
                    <div class="stat-label" style="color: #34D399 !important;">Positive</div>
                    <div class="stat-number" style="color: #34D399 !important;">{pos_cnt}</div>
                    <div class="stat-note">{pos_cnt/total_n*100:.1f}% of total</div>
                </div>
                """, unsafe_allow_html=True)
            with s3:
                st.markdown(f"""
                <div class="stat-box" style="border-color: rgba(244, 63, 94, 0.4);">
                    <div class="stat-label" style="color: #FB7185 !important;">Negative</div>
                    <div class="stat-number" style="color: #FB7185 !important;">{neg_cnt}</div>
                    <div class="stat-note">{neg_cnt/total_n*100:.1f}% of total</div>
                </div>
                """, unsafe_allow_html=True)
            with s4:
                st.markdown(f"""
                <div class="stat-box" style="border-color: rgba(139, 92, 246, 0.4);">
                    <div class="stat-label" style="color: #C084FC !important;">Neutral</div>
                    <div class="stat-number" style="color: #C084FC !important;">{neu_cnt}</div>
                    <div class="stat-note">{neu_cnt/total_n*100:.1f}% of total</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            ch1, ch2 = st.columns(2, gap="medium")
            with ch1:
                fig_donut = px.pie(
                    names=["Positive", "Negative", "Neutral"],
                    values=[pos_cnt, neg_cnt, neu_cnt],
                    hole=0.6,
                    color=["Positive", "Negative", "Neutral"],
                    color_discrete_map={"Positive": "#10B981", "Negative": "#F43F5E", "Neutral": "#8B5CF6"},
                    title="<b>Sentiment Distribution</b>"
                )
                fig_donut.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
                    legend=dict(font=dict(color="#E2E8F0"))
                )
                st.plotly_chart(fig_donut, width="stretch")

            with ch2:
                if "category" in res_df.columns:
                    cat_chart = px.histogram(
                        res_df,
                        x="category",
                        color="Predicted_Sentiment",
                        barmode="group",
                        color_discrete_map={"Positive": "#10B981", "Negative": "#F43F5E", "Neutral": "#8B5CF6"},
                        title="<b>Sentiment by Category</b>"
                    )
                    cat_chart.update_layout(
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#CBD5E1")),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#CBD5E1")),
                        xaxis_title="",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(cat_chart, width="stretch")
                else:
                    conf_hist = px.histogram(
                        res_df,
                        x="Confidence",
                        nbins=20,
                        title="<b>Confidence Score Distribution</b>",
                        color_discrete_sequence=["#3B82F6"]
                    )
                    conf_hist.update_layout(
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#CBD5E1")),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#CBD5E1"))
                    )
                    st.plotly_chart(conf_hist, width="stretch")

            st.markdown("##### Results Table")
            filter_choice = st.multiselect(
                "Filter sentiment:",
                options=["Positive", "Negative", "Neutral"],
                default=["Positive", "Negative", "Neutral"]
            )
            filtered_df = res_df[res_df["Predicted_Sentiment"].isin(filter_choice)]
            st.dataframe(filtered_df, width="stretch", height=280)

            csv_export = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results (CSV)",
                data=csv_export,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )


# =============================================================================
# TAB 3: MODEL EVALUATION
# =============================================================================
with tab3:
    st.markdown("##### Classifier Performance Comparison")
    st.caption("Cross-validated benchmark on 20% stratified test set and 5-fold cross-validation.")

    if metrics_data is not None:
        model_rows = []
        for m_name, m_info in metrics_data["models"].items():
            model_rows.append({
                "Classifier": m_name,
                "Accuracy": f"{m_info['accuracy']*100:.1f}%",
                "Precision": f"{m_info['precision']*100:.1f}%",
                "Recall": f"{m_info['recall']*100:.1f}%",
                "F1-Score": f"{m_info['f1_score']*100:.1f}%",
                "5-Fold CV Score": f"{m_info['cv_mean']*100:.1f}% ± {m_info['cv_std']*100:.1f}%"
            })
        bench_df = pd.DataFrame(model_rows)
        st.dataframe(bench_df, width="stretch")

        st.markdown("---")
        st.markdown("##### Confusion Matrices")
        labels = metrics_data["labels"]
        cm_cols = st.columns(len(metrics_data["models"]))

        # Violet & Blue custom matrix theme
        matrix_colorscale = [
            [0.0, "#0E1019"],
            [0.3, "#2563EB"],
            [1.0, "#8B5CF6"]
        ]

        for i, (m_name, m_info) in enumerate(metrics_data["models"].items()):
            with cm_cols[i]:
                cm_matrix = m_info["confusion_matrix"]
                fig_cm = px.imshow(
                    cm_matrix,
                    labels=dict(x="Predicted", y="True Label", color="Count"),
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale=matrix_colorscale,
                    title=f"<b>{m_name}</b>"
                )
                fig_cm.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
                    xaxis=dict(tickfont=dict(color="#CBD5E1")),
                    yaxis=dict(tickfont=dict(color="#CBD5E1"))
                )
                st.plotly_chart(fig_cm, width="stretch")

        st.markdown("---")
        st.markdown("##### Detailed Classification Metrics")
        sel_rep_model = st.selectbox("Select model:", list(metrics_data["models"].keys()))
        rep_dict = metrics_data["models"][sel_rep_model]["classification_report"]
        
        rep_rows = []
        for lbl in labels:
            if lbl in rep_dict:
                rep_rows.append({
                    "Class": lbl,
                    "Precision": f"{rep_dict[lbl]['precision']*100:.1f}%",
                    "Recall": f"{rep_dict[lbl]['recall']*100:.1f}%",
                    "F1-Score": f"{rep_dict[lbl]['f1-score']*100:.1f}%",
                    "Sample Count": int(rep_dict[lbl]['support'])
                })
        st.dataframe(pd.DataFrame(rep_rows), width="stretch")


# =============================================================================
# TAB 4: DATASET EXPLORER
# =============================================================================
with tab4:
    st.markdown("##### Dataset Overview")
    st.caption("Kaggle multi-domain sentiment dataset (.csv & .json).")

    if dataset_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Total Reviews</div>
                <div class="stat-number">{len(dataset_df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Classes</div>
                <div class="stat-number">{dataset_df['sentiment'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Categories</div>
                <div class="stat-number">{dataset_df['category'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Avg Length</div>
                <div class="stat-number">{int(dataset_df['text'].str.len().mean())} chars</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        d1, d2 = st.columns(2, gap="medium")
        with d1:
            fig_bar = px.histogram(
                dataset_df,
                x="sentiment",
                color="sentiment",
                color_discrete_map={"Positive": "#10B981", "Negative": "#F43F5E", "Neutral": "#8B5CF6"},
                title="<b>Class Distribution</b>"
            )
            fig_bar.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#CBD5E1")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#CBD5E1"))
            )
            st.plotly_chart(fig_bar, width="stretch")

        with d2:
            fig_cat = px.histogram(
                dataset_df,
                x="category",
                color="category",
                color_discrete_sequence=["#8B5CF6", "#3B82F6", "#06B6D4", "#6366F1", "#A855F7", "#2563EB"],
                title="<b>Reviews by Category</b>"
            )
            fig_cat.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#CBD5E1")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#CBD5E1"))
            )
            st.plotly_chart(fig_cat, width="stretch")

        st.markdown("##### Search & Inspect Records")
        s_col1, s_col2 = st.columns([3, 1])
        with s_col1:
            search_kw = st.text_input("Filter by keyword:", placeholder="e.g. movie, service, battery...")
        with s_col2:
            filter_sent = st.multiselect("Class:", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])
        
        disp_df = dataset_df[dataset_df["sentiment"].isin(filter_sent)]
        if search_kw:
            disp_df = disp_df[disp_df["text"].str.contains(search_kw, case=False, na=False)]

        st.dataframe(disp_df, width="stretch", height=280)

        st.markdown("##### Download Files")
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_raw = dataset_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV Dataset", data=csv_raw, file_name="sentiment_dataset.csv", mime="text/csv")
        with dl2:
            with open("data/sentiment_dataset.json", "r", encoding="utf-8") as jf:
                json_raw = jf.read()
            st.download_button("Download JSON Dataset", data=json_raw, file_name="sentiment_dataset.json", mime="application/json")


# =============================================================================
# TAB 5: DOCUMENTATION
# =============================================================================
with tab5:
    st.markdown("##### NLP Methodology & Technical Report")
    st.markdown(r"""
    ### 1. Project Overview
    This project is an end-to-end Sentiment Analysis pipeline built for the Natural Language Processing assignment:
    - **Dataset**: Kaggle benchmark multi-domain sentiment dataset (.csv and .json formats).
    - **Preprocessing**: Contraction expansion, URL/tag cleaning, emoticon normalization, and negation preservation.
    - **Feature Engineering**: Sublinear TF-IDF vectorizer using unigrams and bigrams ($n \in \{1, 2\}$).
    - **Models**: Multinomial Naive Bayes, Logistic Regression (L2 regularized), and Calibrated Linear SVM.
    - **Deployment**: Streamlit web interface with real-time analysis, batch processing, and metric visualization.

    ---

    ### 2. Preprocessing Architecture
    1. **Contraction Expansion**: English contractions (e.g. *don't* $\to$ *do not*, *it's* $\to$ *it is*) are normalized to prevent token splitting artifacts.
    2. **Negation Preservation**: Standard stopword removal strips words like *not*, *no*, and *never*. Removing these reverses sentiment polarity (e.g. *"not good"* becomes *"good"*). This pipeline explicitly preserves all negation terms.
    3. **Sublinear TF-IDF**:
       $$\text{TF-IDF}(t, d, D) = (1 + \log(\text{TF}(t, d))) \times \text{IDF}(t, D)$$
       Logarithmic term-frequency scaling prevents high-frequency repeated words from overshadowing critical sentiment markers.

    ---

    ### 3. Model Formulations
    - **Multinomial Naive Bayes**:
      $$P(c|d) \propto P(c) \prod_{i=1}^n P(w_i|c)$$
    - **Logistic Regression**:
      $$P(y=k|x) = \frac{e^{w_k^T x}}{\sum_j e^{w_j^T x}}$$
    - **Linear Support Vector Machine**:
      $$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \max(0, 1 - y_i(w^T x_i + b))$$
    """)
