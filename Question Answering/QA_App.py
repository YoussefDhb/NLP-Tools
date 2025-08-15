
import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="QA App")
st.title('Question Answering')
# Unique key to avoid duplicate element IDs
model_name = st.sidebar.selectbox('Model', ['distilbert-base-uncased-distilled-squad', 'ktrapeznikov/albert-xlarge-v2-squad-v2'], key='model_select_sidebar')

if st.button('Load model'):
    # Second selectbox must have a different key
    model_name_sel = st.sidebar.selectbox('Model', ['distilbert-base-uncased-distilled-squad', 'ktrapeznikov/albert-xlarge-v2-squad-v2'], key='model_select_loader')
    st.write('Model Loaded')
    device_index = 0 if torch.cuda.is_available() else -1
    pipe = pipeline('question-answering', model=model_name_sel, tokenizer=model_name_sel, device=device_index)
    st.session_state['pipe'] = pipe

context = st.text_area('Context', height=300)
question = st.text_input('Question')
if st.button('Answer') and 'pipe' in st.session_state:
    out = st.session_state['pipe'](question=question, context=context)
    st.write('Answer:', out.get('answer'))
