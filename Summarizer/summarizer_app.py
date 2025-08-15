
import os, re, time, io
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Optional parsers
try:
    import PyPDF2 as _pypdf2
except Exception:
    _pypdf2 = None
try:
    from docx import Document as _DocxDocument
except Exception:
    _DocxDocument = None

st.set_page_config(page_title="Summarizer", page_icon="📝", layout="wide")
st.title("📝 Summarizer App")

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"**Device:** {DEVICE}")

CANDIDATE_MODELS = ["summarizer_ft\\final", "summarizer_ft\\checkpoint-50", "ainize/bart-base-cnn", "google-t5/t5-small"]

# Caching compatibility (Streamlit >=1.18 uses cache_resource)
def _apply_cache(fn):
    if hasattr(st, "cache_resource"):
        return st.cache_resource(show_spinner=True)(fn)
    # Fallback for older versions
    return st.cache(allow_output_mutation=True)(fn)

def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)
    mdl.eval()
    return tok, mdl

load_model = _apply_cache(load_model)

# Safe sentence tokenize (no nltk dependency required at runtime)
def safe_sent_tokenize(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

# Chunking and summarization helpers

def chunk_text(text: str, tokenizer, max_input_tokens: int = 1024, overlap_sentences: int = 1):
    sents = safe_sent_tokenize(text)
    chunks, buf = [], []
    for s in sents:
        candidate = " ".join(buf + [s])
        if len(tokenizer.encode(candidate, truncation=False)) <= max_input_tokens:
            buf.append(s)
        else:
            if buf:
                chunks.append(" ".join(buf))
                buf = buf[-overlap_sentences:] if overlap_sentences > 0 else []
                buf.append(s)
            else:
                tokens = tokenizer.encode(s, truncation=True, max_length=max_input_tokens)
                chunks.append(tokenizer.decode(tokens, skip_special_tokens=True))
                buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def summarize_chunks(chunks, tokenizer, model, summary_max_tokens=128, num_beams=4):
    parts = []
    for ch in chunks:
        inputs = tokenizer(ch, return_tensors="pt", truncation=True, max_length=min(2048, tokenizer.model_max_length)).to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=summary_max_tokens, num_beams=num_beams, early_stopping=True)
        parts.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return " ".join(parts)


def abstractive_summarize(text: str, tokenizer, model, max_input_tokens=1024, summary_max_tokens=128, use_chunking=True, num_beams=4, use_t5_prefix=False):
    if use_t5_prefix:
        text = "summarize: " + text
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) <= max_input_tokens or not use_chunking:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=summary_max_tokens, num_beams=num_beams, early_stopping=True)
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    chunks = chunk_text(text, tokenizer, max_input_tokens=max_input_tokens)
    return summarize_chunks(chunks, tokenizer, model, summary_max_tokens=summary_max_tokens, num_beams=num_beams)

# ---- File upload & extraction ----

def _decode_bytes(data: bytes) -> str:
    if not isinstance(data, (bytes, bytearray)):
        return str(data)
    try:
        return data.decode("utf-8")
    except Exception:
        try:
            return data.decode("latin-1")
        except Exception:
            return ""

def extract_text_from_upload(up_file) -> str:
    name = (up_file.name or "uploaded").lower()
    data = up_file.read()
    bio = io.BytesIO(data)
    if name.endswith(".txt"):
        return _decode_bytes(data)
    if name.endswith(".pdf"):
        if _pypdf2 is None:
            return ""
        try:
            reader = _pypdf2.PdfReader(bio)
            texts = []
            for pg in reader.pages:
                try:
                    t = pg.extract_text() or ""
                    if t:
                        texts.append(t)
                except Exception:
                    pass
            return " ".join(texts).strip()
        except Exception:
            return ""
    if name.endswith(".docx"):
        if _DocxDocument is None:
            return ""
        try:
            doc = _DocxDocument(bio)
            return " ".join([p.text for p in doc.paragraphs if p.text and p.text.strip()]).strip()
        except Exception:
            return ""
    return ""

# Sidebar controls
model_name = st.sidebar.selectbox("Model", options=CANDIDATE_MODELS, index=0)
max_input_tokens = st.sidebar.slider("Max input tokens", 256, 2048, 1024, 64)
summary_max_tokens = st.sidebar.slider("Summary max tokens", 16, 256, 128, 8)
num_beams = st.sidebar.slider("Beam size", 1, 6, 4, 1)
use_chunking = st.sidebar.checkbox("Use chunking for long texts", value=True)
use_t5_prefix = "t5" in model_name.lower()

# Visual cue if a local fine-tuned model is selected
if os.path.isdir(model_name):
    st.sidebar.success("Using local fine-tuned model")

st.sidebar.caption("Tip: T5 models automatically use the 'summarize:' prefix.")

# Backward-compatible primary button

def primary_button(label: str):
    try:
        return st.button(label, type="primary")
    except TypeError:
        return st.button(label)

# File uploader
uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","txt"], accept_multiple_files=True)
combined_text = ""
if uploaded:
    texts = []
    for f in uploaded:
        t = extract_text_from_upload(f)
        if t:
            texts.append(t)
        else:
            nm = f.name or "file"
            if nm.lower().endswith(".pdf") and _pypdf2 is None:
                st.warning("Couldn't read " + nm + ". Install PyPDF2 to enable PDF parsing: pip install PyPDF2")
            elif nm.lower().endswith(".docx") and _DocxDocument is None:
                st.warning("Couldn't read " + nm + ". Install python-docx to enable DOCX parsing: pip install python-docx")
            else:
                st.warning("Couldn't extract text from " + nm)
    if texts:
        combined_text = "  ".join(texts)
        st.success("Loaded " + str(len(texts)) + " file(s)")
        if not st.session_state.get("input_text"):
            st.session_state["input_text"] = combined_text[:200000]

# Main input area
text = st.text_area("Enter article text", height=300, placeholder="Paste a long article here…", key="input_text")

if primary_button("Summarize"):
    if not text or not text.strip():
        st.warning("Please paste some text or upload a file.")
    else:
        with st.spinner("Loading model and generating…"):
            tok, mdl = load_model(model_name)
            t0 = time.time()
            out = abstractive_summarize(
                text.strip(), tok, mdl,
                max_input_tokens=max_input_tokens,
                summary_max_tokens=summary_max_tokens,
                use_chunking=use_chunking,
                num_beams=num_beams,
                use_t5_prefix=use_t5_prefix,
            )
            dt = time.time() - t0
        st.success(f"Done in {dt:.2f}s")
        st.subheader("Summary")
        st.write(out)
