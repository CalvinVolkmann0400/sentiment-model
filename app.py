import streamlit as st
from transformers import pipeline

LABELS = ["ready to buy now", "just exploring", "can't afford it"]


@st.cache_resource(show_spinner="Loading model...")
def load_classifier():
    """Load the zero-shot classification model once."""
    return pipeline("zero-shot-classification",
                    model="facebook/bart-large-mnli")


st.title("sunsettr Lead Qualifier")
classifier = load_classifier()
answer = st.text_input("Enter a lead's answer")
if answer:
    try:
        result = classifier(answer, candidate_labels=LABELS)
        st.write(f"{result['labels'][0]} ({result['scores'][0]:.0%})")
    except Exception as e:
        st.error(f"Couldn't classify that answer: {e}. Please try again.")
