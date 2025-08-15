
import os, re, io, string, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional readers
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    from docx import Document  # python-docx
except Exception:
    Document = None

# ========== Load Artifacts ==========
ARTS = Path(__file__).parent / "sentiment_artifacts"
vectorizer = joblib.load(ARTS / "vectorizer.joblib")
lr = joblib.load(ARTS / "logreg.joblib")
nb = joblib.load(ARTS / "naive_bayes.joblib")

# ========== Cleaning (mirrors training) ==========
USE_SPACY = True
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        nlp = spacy.blank("en")
except Exception:
    USE_SPACY = False
    nlp = None
    warnings.warn("spaCy unavailable; falling back to NLTK where possible.")

# NLTK stopwords + WordNet (with graceful fallback)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet'); nltk.download('omw-1.4')
    stop_words = set(stopwords.words("english"))
    negation_keep = {"no", "nor", "not", "n't", "never"}
    stop_words = {w for w in stop_words if w not in negation_keep}
    wnl = WordNetLemmatizer()
except Exception:
    stop_words = set()
    wnl = None

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

def lemmatize_tokens(tokens):
    if nlp:
        doc = nlp(" ".join(tokens))
        return [t.lemma_ for t in doc]
    if wnl:
        return [wnl.lemmatize(t) for t in tokens]
    return tokens

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)
    tokens = [tok for tok in text.split() if tok.isalpha()]
    tokens = lemmatize_tokens(tokens)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

# ========== File readers ==========

def read_txt(file) -> str:
    try:
        data = file.read()
        if isinstance(data, bytes):
            return data.decode('utf-8', errors='ignore')
        return str(data)
    finally:
        try:
            file.seek(0)
        except Exception:
            pass

def read_pdf(file) -> str:
    # Try pdfplumber first
    try:
        if pdfplumber is not None:
            with pdfplumber.open(file) as pdf:
                pages_text = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages_text)
    except Exception:
        try:
            file.seek(0)
        except Exception:
            pass
    # Fallback to PyPDF2
    try:
        if PdfReader is not None:
            reader = PdfReader(file)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n".join(pages)
    except Exception:
        pass
    return ""

def read_docx(file) -> str:
    if Document is None:
        return ""
    try:
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

# ========== Inference ==========

def predict_both(text: str):
    x = vectorizer.transform([clean_text(text)])
    prob_lr = float(lr.predict_proba(x)[0][1])
    prob_nb = float(nb.predict_proba(x)[0][1])
    return prob_lr, prob_nb

# ========== UI ==========

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")
st.title("Sentiment Analysis")

with st.sidebar:
    st.header("Settings")
    compare = st.checkbox("Compare both models", value=True)
    model_choice = st.radio("Model", ["Logistic Regression", "Naive Bayes"], index=0, disabled=compare)
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

single_tab, batch_tab, files_tab = st.tabs(["Single Text", "Batch CSV", "Files (PDF/TXT/DOCX)"]) 

with single_tab:
    txt = st.text_area("Enter a review", height=160, placeholder="Type or paste a product/movie review…")
    if st.button("Predict Sentiment", type="primary", use_container_width=True):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            p_lr, p_nb = predict_both(txt)
            if compare:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Logistic Regression")
                    st.write(f"Positive: {p_lr:.3f} — Negative: {1-p_lr:.3f}")
                    st.progress(p_lr)
                    st.write("Label:", "Positive" if p_lr >= threshold else "Negative")
                with col2:
                    st.subheader("Naive Bayes")
                    st.write(f"Positive: {p_nb:.3f} — Negative: {1-p_nb:.3f}")
                    st.progress(p_nb)
                    st.write("Label:", "Positive" if p_nb >= threshold else "Negative")
            else:
                if model_choice == "Logistic Regression":
                    label = "Positive" if p_lr >= threshold else "Negative"
                    st.subheader(f"Prediction: {label}")
                    st.write(f"Probability Positive: {p_lr:.3f} — Negative: {1-p_lr:.3f}")
                    st.progress(p_lr)
                else:
                    label = "Positive" if p_nb >= threshold else "Negative"
                    st.subheader(f"Prediction: {label}")
                    st.write(f"Probability Positive: {p_nb:.3f} — Negative: {1-p_nb:.3f}")
                    st.progress(p_nb)

with batch_tab:
    st.write("Upload a CSV with a column named 'review' (or 'text'/'review_text').")
    file = st.file_uploader("CSV file", type=["csv"]) 
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None
        if df is not None:
            cand_cols = [c for c in df.columns if c.lower() in ["review","text","review_text","content","body"]]
            if not cand_cols:
                st.error("No review text column found. Add a 'review' column.")
            else:
                col = cand_cols[0]
                cleaned = df[col].astype(str).apply(clean_text)
                X = vectorizer.transform(cleaned)
                proba_lr = lr.predict_proba(X)[:, 1]
                proba_nb = nb.predict_proba(X)[:, 1]
                if compare:
                    out = df.copy()
                    out["LR_proba_pos"] = proba_lr
                    out["LR_label"] = np.where(out["LR_proba_pos"]>=threshold, "Positive", "Negative")
                    out["NB_proba_pos"] = proba_nb
                    out["NB_label"] = np.where(out["NB_proba_pos"]>=threshold, "Positive", "Negative")
                    st.write("Preview:")
                    st.dataframe(out.head(20))
                    st.write("Counts (LR):", out["LR_label"].value_counts())
                    st.write("Counts (NB):", out["NB_label"].value_counts())
                    st.download_button(
                        "Download predictions CSV",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="sentiment_predictions_compare.csv",
                        mime="text/csv",
                    )
                else:
                    clf = lr if model_choice == "Logistic Regression" else nb
                    proba = proba_lr if model_choice == "Logistic Regression" else proba_nb
                    pred = (proba >= threshold).astype(int)
                    out = df.copy()
                    out["pred_label"] = np.where(pred==1, "Positive", "Negative")
                    out["pred_proba_pos"] = proba
                    st.write("Preview:")
                    st.dataframe(out.head(20))
                    st.write("Counts:", out["pred_label"].value_counts())
                    st.download_button(
                        "Download predictions CSV",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )

with files_tab:
    st.write("Upload one or more files: PDF, TXT, or DOCX.")
    files = st.file_uploader("Files", type=["pdf","txt","docx"], accept_multiple_files=True)
    if files:
        rows = []
        for f in files:
            name = f.name
            suffix = Path(name).suffix.lower()
            try:
                if suffix == ".pdf":
                    text = read_pdf(f)
                elif suffix == ".txt":
                    text = read_txt(f)
                elif suffix == ".docx":
                    text = read_docx(f)
                else:
                    text = ""
            except Exception:
                text = ""
            if not text.strip():
                if compare:
                    rows.append({"file": name, "LR_label": "(no text)", "LR_proba_pos": np.nan, "NB_label": "(no text)", "NB_proba_pos": np.nan, "chars": 0})
                else:
                    rows.append({"file": name, "pred_label": "(no text)", "pred_proba_pos": np.nan, "chars": 0})
                continue
            p_lr, p_nb = predict_both(text)
            if compare:
                rows.append({"file": name, "LR_label": ("Positive" if p_lr>=threshold else "Negative"), "LR_proba_pos": p_lr, "NB_label": ("Positive" if p_nb>=threshold else "Negative"), "NB_proba_pos": p_nb, "chars": len(text)})
            else:
                if model_choice == "Logistic Regression":
                    rows.append({"file": name, "pred_label": ("Positive" if p_lr>=threshold else "Negative"), "pred_proba_pos": p_lr, "chars": len(text)})
                else:
                    rows.append({"file": name, "pred_label": ("Positive" if p_nb>=threshold else "Negative"), "pred_proba_pos": p_nb, "chars": len(text)})
        res = pd.DataFrame(rows)
        st.write("Results:")
        st.dataframe(res)
        if compare:
            st.download_button(
                "Download results CSV",
                res.to_csv(index=False).encode("utf-8"),
                file_name="file_sentiment_predictions_compare.csv",
                mime="text/csv",
            )
        else:
            st.download_button(
                "Download results CSV",
                res.to_csv(index=False).encode("utf-8"),
                file_name="file_sentiment_predictions.csv",
                mime="text/csv",
            )

st.caption("Tip: run this app from a terminal: streamlit run sentiment_analysis.py")
