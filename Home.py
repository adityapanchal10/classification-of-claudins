import streamlit as st

from core.ui import app_header, global_sidebar

st.set_page_config(page_title="Functional Classification of Claudins", layout="wide", page_icon="🧬")
st.logo("🧬")
app_header()
global_sidebar()

st.markdown(
    """
### Overview
Classify claudin sequences using **ESM MSA-1b** and **ESM2** embeddings with a family of trained classifiers.
Explore what drives the model's predictions with per-residue importance scores and attention maps, and visualise them in the context of the sequence and predicted structure.

| Page | What it does |
|---|---|
| 🔮 **Predict** | Batch inference · per-residue IG and attention/saliency heatmaps · ESMFold structure prediction |
| ⚖️ **Compare Models** | Side-by-side prediction and normalised bar-chart explainability for two models |
| 📊 **Data Exploration** | PCA embedding distributions with interactive filtering |
| ℹ️ **About Models** | Architecture summaries and training checkpoint metrics for all registered models |

Use the **sidebar** to set the active model and IG step count before running inference.

**MSA mode toggle**: The sidebar includes an "Embed in MSA mode" toggle. When enabled, embeddings are generated with full MSA context; when disabled, sequences are embedded independently. This toggle is disabled automatically when ESM2 is selected.

**ECS-only mode**: Toggle "ECS only" on the Predict or Compare pages and set ECS1/ECS2 ranges (1-based, inclusive). The app still embeds the full MSA, then slices to the ECS regions for inference; explainability plots are shown on the full sequence with non-ECS positions set to zero.
"""
)

st.info("Start with the **Predict** page in the sidebar.")
