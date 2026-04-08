import streamlit as st

from core.ui import app_header, global_sidebar

st.set_page_config(page_title="Functional Classification of Claudins", layout="wide", page_icon="🧬")
st.logo("🧬")
app_header()
global_sidebar()

st.markdown(
    """
### Overview
Classify claudin sequences using **ESM MSA-1b** embeddings and a family of trained classifiers.
Explore what drives the model's predictions with per-residue importance scores and attention maps, and visualise them in the context of the sequence and predicted structure.

| Page | What it does |
|---|---|
| 🔮 **Predict** | Batch inference · per-residue IG and attention heatmaps · ESMFold structure prediction |
| ⚖️ **Compare Models** | Side-by-side prediction and normalised bar-chart explainability for two models |
| 📊 **Data Exploration** | PCA embedding distributions with interactive filtering |
| ℹ️ **About Models** | Architecture summaries and training checkpoint metrics for all registered models |

Use the **sidebar** to set the active model and IG step count before running inference.
"""
)

st.info("Start with the **Predict** page in the sidebar.")
