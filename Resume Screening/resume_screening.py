
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, io
import numpy as np

st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="wide")
st.title('Resume Screening')

# Model loading (adjust path if running in separate env)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.sidebar.header('Job selector')
# Prefer datasets/resumejobs/job_descriptions.csv, else fallback to jobs.csv
jobs = None
job_text = ''

def normalize_text(text: str):
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    return re.sub(r'\s+', ' ', text.replace('\r',' ').replace('\n',' ')).strip()

def load_jobs_df():
    # Try dataset path first
    ds_path = os.path.join('datasets','resumejobs','job_descriptions.csv')
    if os.path.exists(ds_path):
        df = pd.read_csv(ds_path, nrows=5000)  # cap for snappy UI
    elif os.path.exists('jobs.csv'):
        df = pd.read_csv('jobs.csv')
    else:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    def pick(keys):
        for k in keys:
            if k in lower_map:
                return lower_map[k]
        return None
    desc_col = pick(['job description','description','job_description','jd','desc','text','summary','requirements','qualifications']) or df.columns[0]
    title_col = pick(['job title','title','job_title','position','role'])
    id_col = pick(['job id','job_id','jobid','id'])
    rename_map = {desc_col: 'description'}
    if title_col: rename_map[title_col] = 'title'
    if id_col: rename_map[id_col] = 'job_id'
    df = df.rename(columns=rename_map)
    if 'title' not in df.columns:
        df['title'] = df.index.astype(str)
    if 'job_id' not in df.columns:
        df['job_id'] = df.index.astype(str)
    df['description_clean'] = df['description'].astype(str).apply(normalize_text)
    return df

# Lightweight skill matching
@st.cache_resource
def load_skills():
    skills = []
    if os.path.exists('skills.txt'):
        with open('skills.txt','r',encoding='utf-8',errors='ignore') as f:
            skills = [l.strip() for l in f if l.strip()]
    return [s.lower() for s in skills]

skills_list = load_skills()

jobs = load_jobs_df()
if jobs is not None and len(jobs) > 0:
    job_idx = st.sidebar.selectbox(
        'Choose job',
        jobs.index.tolist(),
        format_func=lambda i: f"{jobs.loc[i,'title']} (id={jobs.loc[i,'job_id']})" if 'job_id' in jobs.columns else f"{jobs.loc[i,'title']} (row {i})"
    )
    job_text = jobs.loc[job_idx, 'description_clean']
else:
    st.sidebar.info('Place datasets/resumejobs/job_descriptions.csv or jobs.csv in the app folder.')
    job_text = st.text_area('Or paste a job description here:')

# Sidebar options
st.sidebar.header('Options')
top_k = st.sidebar.slider('Top-K resumes to display', 1, 50, 10)
show_just = st.sidebar.checkbox('Show justifications for top matches', value=False)
just_n = st.sidebar.slider('Sentences per resume (when showing justification)', 1, 5, 3)

# Readers for uploaded files
def read_txt_file(uploaded):
    data = uploaded.read()
    try:
        return data.decode('utf-8', errors='ignore') if isinstance(data, (bytes, bytearray)) else str(data)
    finally:
        try:
            uploaded.seek(0)
        except Exception:
            pass

def read_pdf_file(uploaded):
    text = ''
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ''
    except Exception:
        try:
            uploaded.seek(0)
        except Exception:
            pass
    finally:
        try:
            uploaded.seek(0)
        except Exception:
            pass
    return text

def read_docx_file(uploaded):
    text = ''
    try:
        import docx
        doc = docx.Document(io.BytesIO(uploaded.read()))
        text = '\n'.join([p.text for p in doc.paragraphs])
    except Exception:
        pass
    finally:
        try:
            uploaded.seek(0)
        except Exception:
            pass
    return text

def extract_skills_simple(text_lower: str):
    if not skills_list:
        return []
    found = []
    for s in skills_list:
        if s and s in text_lower:
            found.append(s)
    # dedupe preserve order
    seen = set()
    out = []
    for s in found:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

# Inference helpers
def get_score(job_text: str, resume_text: str) -> float:
    if not job_text or not resume_text:
        return 0.0
    job_emb = model.encode([job_text], convert_to_numpy=True)
    resume_emb = model.encode([resume_text], convert_to_numpy=True)
    return float(cosine_similarity(job_emb, resume_emb).squeeze())

def justify(job_text: str, resume_text: str, k: int = 3):
    job_sents = [s.strip() for s in re.split(r'[\n\.!?]+', job_text) if s.strip()][:200]
    res_sents = [s.strip() for s in re.split(r'[\n\.!?]+', resume_text) if s.strip()][:200]
    if not job_sents or not res_sents:
        return []
    ej = model.encode(job_sents, convert_to_numpy=True)
    er = model.encode(res_sents, convert_to_numpy=True)
    sim = cosine_similarity(er, ej).max(axis=1)
    idx = np.argsort(sim)[-k:][::-1]
    return [res_sents[i] for i in idx]

uploaded_files = st.file_uploader('Upload resumes (.pdf, .docx, .txt) — multiple allowed', type=['pdf','docx','txt'], accept_multiple_files=True)

if not job_text:
    st.warning('Provide or select a job description first.')
else:
    if uploaded_files:
        rows = []
        for uf in uploaded_files:
            name = uf.name
            ext = name.split('.')[-1].lower()
            if ext == 'txt':
                resume_text = read_txt_file(uf)
            elif ext == 'pdf':
                resume_text = read_pdf_file(uf)
            elif ext == 'docx':
                resume_text = read_docx_file(uf)
            else:
                resume_text = ''
            text_clean = normalize_text(resume_text)
            score = get_score(job_text, text_clean)
            matched = extract_skills_simple(text_clean.lower()) if text_clean else []
            rows.append({
                'file': name,
                'chars': len(text_clean),
                'matching score': round(score*100, 2),
                'text': text_clean,
            })
        res = pd.DataFrame(rows).sort_values('matching score', ascending=False).reset_index(drop=True)
        st.subheader('Top matches')
        st.dataframe(res[['file','chars','matching score']].head(top_k), use_container_width=True)
        st.download_button('Download results CSV', res.to_csv(index=False).encode('utf-8'), file_name='resume_screening_results.csv', mime='text/csv')

        if show_just:
            st.markdown('---')
            st.subheader('Justifications')
            top_subset = res.head(top_k)
            for i, row in top_subset.iterrows():
                with st.expander(f"{row['file']} — Matching Score: {row['matching score']}%"):
                    sents = justify(job_text, row['text'], k=just_n)
                    if sents:
                        for s in sents:
                            st.write('- ', s)
                    else:
                        st.write('No sentences found.')
    else:
        st.info('Upload one or more resumes to score against the selected job.')

