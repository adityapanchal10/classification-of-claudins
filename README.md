# Functional Classification of Claudins — Streamlit App

A multi-page Streamlit application for classifying protein sequences using MSA Transformer (ESM-MSA-1b) embeddings and a family of trained classifiers. The app supports batch inference, per-residue explainability, side-by-side model comparison, and embedding distribution exploration.

---

## Pages

| Page | Purpose |
|---|---|
| 🔮 **Predict** | Run inference on one or more sequences. Inspect any sequence with Integrated Gradients (IG) and attention heatmaps. Predict structure through the public ESMFold API and color it by residue importance. Results persist in session state for use in other pages. |
| ⚖️ **Compare Models** | Select two models and compare their predictions and explainability side-by-side for any sequence already run on the Predict page (or new input). Uses normalised per-residue bar charts for IG scores and attention weights. |
| 📊 **Data Exploration** | Visualise per-residue embedding distributions using PCA or raw-dimension analysis. Accepts pre-stored embeddings from the Predict page or a separately uploaded NPY/PT file. Filter sequences via a multiselect widget with sticky frosted-glass controls. |
| ℹ️ **About Models** | Overview table of all registered models. Shows a `torchinfo` architecture summary and training checkpoint metrics (epoch, validation AUC, accuracy, F1) for the selected model. |

---

## Classification Task

Three-class channel-protein classification:

| Label | Class |
|---|---|
| 0 | Barrier forming |
| 1 | Cation-channel forming |
| 2 | Anion-channel forming |

---

## Registered Models

| Model | Architecture summary | Attention |
|---|---|---|
| **Transformer + MLP Classifier** | Linear projection → positional embedding → self-attention blocks → attention/mean/max pooling → fusion MLP → linear head | ✅ |
| **Simple Linear Classifier** | LayerNorm → learned attention scores → softmax-weighted sum → dropout → linear head | ❌ |
| **Simple CNN Classifier** | LayerNorm → parallel Conv2d (kernels 3/4/5) → ReLU → global max pooling → concat → dropout → linear head | ❌ |
| **Transformer Classifier (simple)** | Positional embedding add → TransformerEncoder → mean pooling → 2-layer MLP head | ✅ |
| **Transformer Classifier (complex)** | Input projection → positional embedding → residual Conv1d blocks → attention pooling → MLP head | ✅ |

Checkpoints live in `checkpoints/`. Each `.pt` file stores the model weights and optionally training metrics (`epoch`, `val_auc`, `val_acc`, `val_f1`).

---

## Project Structure

```
app.py                           # Landing page, sidebar state initialisation
requirements.txt
checkpoints/                    # Model checkpoint files (.pt) and ESM alphabet
pages/
    1_🔮_Predict.py             # Inference, explainability, structure prediction
    2_⚖️_Compare_Models.py      # Side-by-side model comparison
    3_📊_Data_Exploration.py    # PCA / raw-dim embedding visualisation
    4_ℹ️_About_Models.py        # Model registry overview and checkpoint stats
core/
    config.py                   # CLASS_MAP, MODEL_REGISTRY, path constants
    models.py                   # Classifier architectures and checkpoint loading
    io_utils.py                 # FASTA / plain-text parsing and sequence validation
    embeddings.py               # ESM-MSA-1b per-residue embedder, ESMFold API helper
    predict.py                  # predict_probabilities(), build_prediction_table()
    explainability.py           # Integrated Gradients, attention and IG dataframes
    visuals.py                  # Plotly charts: heatmaps, bar charts, PCA plots, structure viewer
    ui.py                       # global_sidebar(), app_header(), shared defaults
```

---

## Session State Keys

Pages share data through `st.session_state`:

| Key | Set by | Used by |
|---|---|---|
| `input_sequences_df` | Predict | Compare Models, Data Exploration |
| `generated_embeddings` | Predict | Compare Models, Data Exploration |
| `predict_run` | Predict | Compare Models (pre-selects inspected sequence) |
| `global_model_name` | Sidebar / any page | All pages |
| `global_ig_steps` | Sidebar | Predict, Compare Models |

---

## Visualisations

| Chart | Where | Details |
|---|---|---|
| Residue heatmap (IG / attention) | Predict | Fixed 13 px cells, horizontal scroll, drag pan, double-click 1.5× zoom, fixed frosted-glass colorbar |
| Per-residue bar chart (IG / attention) | Compare Models | Normalised to [−1, 1] or [0, 1]; diverging RdBu for IG, Blues for attention; theme-aware |
| PCA residue boxplots + heatmap | Data Exploration | One box per sequence per residue; explained-variance table; theme-aware diverging heatmap |
| Raw-dim distribution boxplots + mean heatmap | Data Exploration | Full embedding-dimension spread per residue |
| Sequence summary scatter | Data Exploration | Mean norm vs. mean spread across sequences |
| 3-D structure viewer | Predict | ESMFold API structure fetch; py3Dmol cartoon rendering colored by residue importance; PDB download |

All Plotly charts switch colorscale automatically when the Streamlit theme changes (light ↔ dark). Data Exploration calls `st.rerun()` on theme change to refresh charts immediately.

---

## Running the App

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

The ESM MSA-1b model weights are downloaded automatically on first run via the `esm` library.  
ESMFold structure prediction is fetched from the public API at `https://api.esmatlas.com/foldSequence/v1/pdb/` and is optional. Residue importance from IG is written into the structure viewer's B-factors for coloring.

---

## Extending the App

### Add a new classifier model

1. Define the architecture class in `core/models.py`.
2. Save a trained checkpoint to `checkpoints/`.
3. Add an entry to `MODEL_REGISTRY` in `core/config.py` — the rest of the app picks it up automatically.

### Add a new class label

Update `CLASS_MAP` in `core/config.py`. All prediction tables and explainability logic derive labels from this mapping.
