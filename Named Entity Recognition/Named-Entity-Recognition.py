import warnings
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy
import streamlit as st

st.set_page_config(page_title="NER App", layout="wide")

CANDIDATES = ["en_core_web_md", "en_core_web_sm"]


def build_entity_ruler(nlp, overwrite_ents=False):
    if "entity_ruler_custom" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler_custom")
    ruler = nlp.add_pipe(
        "entity_ruler",
        name="entity_ruler_custom",
        last=True,
        config={"overwrite_ents": overwrite_ents},
    )
    patterns = []
    org_suffixes = [
        "Inc",
        "Inc.",
        "Corp",
        "Corp.",
        "Corporation",
        "LLC",
        "Ltd",
        "Ltd.",
        "University",
        "Institute",
        "Group",
        "Co.",
        "Company",
    ]
    for suf in org_suffixes:
        patterns.append({"label": "ORG", "pattern": [{"IS_TITLE": True, "OP": "+"}, {"TEXT": suf}]})
    titles = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "President", "Prime Minister", "Sir"]
    for t in titles:
        patterns.append({"label": "PERSON", "pattern": [{"TEXT": t}, {"IS_TITLE": True, "OP": "+"}]})
    gpes = [
        "United States",
        "U.S.",
        "USA",
        "United Kingdom",
        "UK",
        "New York",
        "London",
        "Paris",
        "Berlin",
        "Tokyo",
    ]
    for g in gpes:
        patterns.append({"label": "GPE", "pattern": g})
    orgs = ["United Nations", "European Union", "Apple Inc", "Google", "Microsoft", "Facebook", "Twitter"]
    for o in orgs:
        patterns.append({"label": "ORG", "pattern": o})
    ruler.add_patterns(patterns)
    return ruler


@st.cache_resource(show_spinner=False)
def load_models():
    loaded = {}
    # Load candidate spaCy models if available
    for name in CANDIDATES:
        try:
            loaded[name] = spacy.load(name)
        except Exception:
            pass

    # Select a base model for the rule-based pipeline
    base = None
    base_name = None
    if loaded:
        base_name, base = next(iter(loaded.items()))
    else:
        base_name = "blank_en"
        base = spacy.blank("en")
        warnings.warn("Using spacy.blank('en') because no pre-trained models were found.")

    # Build an independent ruler model
    try:
        ruler_model = base.copy()
    except Exception:
        try:
            ruler_model = spacy.load(base_name) if base_name != "blank_en" else spacy.blank("en")
        except Exception:
            ruler_model = spacy.blank("en")

    build_entity_ruler(ruler_model, overwrite_ents=False)
    loaded = {"rule_based": ruler_model, **loaded}
    return loaded


models = load_models()
model_names = list(models.keys())

st.title("Named Entity Recognition")
st.write(
    "Type or paste text and compare NER models, including a rule-based EntityRuler."
)

col1, col2 = st.columns([3, 1])
with col2:
    model_choice = st.selectbox("Model", model_names, index=0)
    max_chars = st.slider("Max text length", 100, 5000, 1000, 100)
with col1:
    default_text = (
        "Apple Inc. hired Dr. John Smith in New York to lead AI research for the European Union."
    )
    text = st.text_area("Input text", default_text, height=200, max_chars=max_chars)

if st.button("Analyze"):
    nlp = models[model_choice]
    doc = nlp(text)
    ents = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
    df = pd.DataFrame(ents)

    if df.empty:
        st.info("No entities found.")
    else:
        labels = sorted(df["label"].unique().tolist())
        selected = st.multiselect("Filter labels", labels, default=labels)
        st.subheader("Entities")
        st.dataframe(df[df["label"].isin(selected)], use_container_width=True)

    # Render highlighted entities
    html = displacy.render(doc, style="ent", options={"compact": True})
    st.subheader("Highlighted text")
    st.components.v1.html(html, height=250, scrolling=True)