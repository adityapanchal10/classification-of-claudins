import streamlit as st

from core.config import MODEL_REGISTRY


DEFAULT_SEQ_LENGTH = 190
DEFAULT_BATCH_SIZE = 64
DEFAULT_IG_STEPS = 50


def app_header():
    st.title("Claudin Classification and Explainability with MSA Transformer Embeddings")
    st.caption("Pretrained embedding -> model selection -> prediction -> explainability -> structure")


def global_sidebar():
    model_options = list(MODEL_REGISTRY.keys())
    default_model = model_options[0]
    if st.session_state.get("global_model_name") not in model_options:
        st.session_state["global_model_name"] = default_model
    if "global_ig_steps" not in st.session_state:
        st.session_state["global_ig_steps"] = DEFAULT_IG_STEPS

    st.sidebar.header("Global settings")
    model_name = st.sidebar.selectbox("Model", model_options, key="global_model_name")
    ig_steps = st.sidebar.slider("Integrated Gradients steps", min_value=50, max_value=200, step=10, key="global_ig_steps")
    return model_name, DEFAULT_SEQ_LENGTH, DEFAULT_BATCH_SIZE, ig_steps
