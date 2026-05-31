# Functional Classification of Claudins — Streamlit App

A multi-page Streamlit application for classifying claudin sequences using ESM embeddings from the MSA Transformer (ESM-MSA-1b) and ESM2, together with a family of trained classifiers. The app supports batch inference, per-residue explainability, side-by-side model comparison, and embedding distribution exploration.

---

## Pages

| Page | Purpose |
|---|---|
| 🔮 **Predict** | Run inference on one or more sequences. Inspect any sequence with Integrated Gradients (IG) and attention heatmaps (for models using attention). Predict structure through the public ESMFold API and color it by residue importance. Results persist in session state for use in other pages. |
| ⚖️ **Compare Models** | Select two models and compare their predictions and explainability side-by-side for any sequence already run on the Predict page (or new input). Uses normalised per-residue bar charts for IG scores and attention weights. |
| 📊 **Data Exploration** | Visualise per-residue embedding distributions using PCA. Uses pre-stored embeddings when available, or generates embeddings from the provided input sequences. Filter sequences via a multiselect widget with sticky frosted-glass controls and run exploration on demand. |
| ℹ️ **About Models** | Overview table of all registered models. Shows a `torchinfo` architecture summary and training checkpoint metrics (saved epoch, validation AUC, accuracy, loss, and % class error curves) for the selected model. |

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
| **Transformer + MLP** | Linear projection → positional embedding → self-attention blocks → attention/mean/max pooling → fusion MLP → linear head | ✅ |
| **Transformer + MLP 2** | Linear projection → positional embedding → self-attention blocks → attention/mean/max pooling → fusion MLP → linear head | ✅ |
| **Transformer + MLP 2 (ECS only)** | Linear projection → positional embedding → self-attention blocks → attention/mean/max pooling → fusion MLP → linear head | ✅ |
| **Transformer + MLP 2 ESM2** | Linear projection → positional embedding → self-attention blocks → attention/mean/max pooling → fusion MLP → linear head (trained on ESM2 embeddings) | ✅ |
| **Transformer + MLP 2 ESM2 (ECS only)** | Linear projection → positional embedding → self-attention blocks → attention/mean/max pooling → fusion MLP → linear head (trained on ESM2 embeddings, ECS-only) | ✅ |
| **Simple Linear** | LayerNorm → learned attention scores → softmax-weighted sum → dropout → linear head | ❌ |
| **Simple CNN** | LayerNorm → parallel Conv2d (kernels 3/4/5) → ReLU → global max pooling → concat → dropout → linear head | ❌ |
| **Transformer (simple)** | Positional embedding add → TransformerEncoder → mean pooling → 2-layer MLP head | ✅ |
| **Transformer (complex)** | Input projection → positional embedding → residual Conv1d blocks → attention pooling → MLP head | ✅ |

Checkpoints live in `checkpoints/`. Each `.pt` file stores the model weights and optionally training metrics (`epoch`, `val_auc`, `acc`, `loss`, and `% class errors`).

Note: Models suffixed with `ESM2` were trained on ESM2 (640-d) embeddings and therefore require the `ESM2` embedder to be selected. The app filters model lists by the selected embedder to prevent mismatched pairings.

---

## Project Structure

```
Home.py                         # Landing page, sidebar state initialisation
requirements.txt
checkpoints/                    # Model checkpoint files (.pt) and ESM alphabet
pages/
    1_🔮_Predict.py             # Inference, explainability, structure prediction
    2_⚖️_Compare_Models.py      # Side-by-side model comparison
    3_📊_Data_Exploration.py    # PCA embedding visualisation
    4_ℹ️_About_Models.py        # Model registry overview and checkpoint stats
`core/`
    config.py                   # CLASS_MAP, MODEL_REGISTRY, path constants
    models.py                   # Classifier architectures and checkpoint loading
    io_utils.py                 # FASTA / plain-text parsing and sequence validation
    embeddings.py               # ESM-MSA-1b / ESM2 per-residue embedder, ESMFold API helper
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
| `generated_embeddings_embedder` | Predict | Tracks which embedder was used for `generated_embeddings` |
| `generated_embeddings_msa_only` | Predict | Tracks whether the cached `generated_embeddings` were produced in MSA mode |
| `predict_run` | Predict | Compare Models (pre-selects inspected sequence) |
| `global_model_name` | Sidebar / any page | All pages |
| `global_ig_steps` | Sidebar | Predict, Compare Models |

---

## Visualisations

| Chart | Where | Details |
|---|---|---|
| Residue heatmap (IG / attention) | Predict | Fixed 13 px cells, horizontal scroll, drag pan, double-click 1.5× zoom, fixed transparent colorbar |
| Per-residue bar chart (IG / attention) | Compare Models | Normalised to [−1, 1] or [0, 1]; diverging RdBu for IG, Blues for attention; theme-aware |
| PCA residue boxplots + heatmap | Data Exploration | One box per sequence per residue; explained-variance table; theme-aware diverging heatmap |
| Sequence summary scatter | Data Exploration | Mean norm vs. mean spread across sequences |
| 3-D structure viewer | Predict | ESMFold API structure fetch; py3Dmol rendering colored by residue contribution to the prediction; PDB download |

---

## Running the App

Try out the app here: https://classification-of-claudins.streamlit.app/ 

**OR**

if you want to run locally:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run Home.py
```

The ESM MSA-1b and ESM2 model weights are downloaded automatically on first run via the `esm` library.  
The trained classifier weights are stored on `GDrive` and fetched from there. Once fetched, all weights are stored locally and re-used.
ESMFold structure prediction is fetched from the public API at `https://api.esmatlas.com/foldSequence/v1/pdb/` and is optional. Residue importance from IG is written into the structure viewer's B-factors for coloring.

---

## ECS-Only Mode

Predict and Compare pages include an **ECS only** toggle. When enabled, you provide ECS1 and ECS2 ranges (1-based, inclusive). The app still embeds the full MSA, then slices out only the ECS regions for inference. Explainability plots remain on the full sequence, with non-ECS positions set to zero so only ECS residues are highlighted.

---

## MSA Mode Toggle (applicable only for the 'MSA Transformer' Embedder)

The sidebar includes an **Embed in MSA mode** toggle. When enabled, embeddings are generated using the full MSA context; when disabled, the embedder treats each sequence independently. This toggle is disabled automatically when the ESM2 embedder is selected. In Compare Models, each model has its own MSA toggle so you can compare MSA-on vs MSA-off behavior side by side.

---

## Embedder Options and Compatibility

- **MSA Transformer (ESM-MSA-1b)**: supports MSA mode and produces 768-dimensional per-residue embeddings. Token handling for this model removes the leading BOS token when converting model outputs to per-residue embeddings.
- **ESM2**: does not support MSA mode in this app and produces 640-dimensional per-residue embeddings. ESM2 appends both BOS and EOS tokens to sequences; the app removes both when producing per-residue embeddings.

Important compatibility notes:
- Classifier checkpoints are tied to an embedding dimensionality. 
- The sidebar model dropdown is filtered to show only models explicitly marked as compatible with the currently selected embedder (fallback to showing all models if none are marked compatible). This prevents accidental mismatches.
- The Compare page exposes per-column embedder selectors (Embedder A / Embedder B). Each column's Model dropdown is filtered to models compatible with that column's selected embedder; pre-stored embeddings from the Predict page are only reused when the embedder name and MSA-mode match.

---

## Interpreting Attention and Saliency

- **Attention**: higher values mean the model is focusing more on that residue. The scale is low → high.
- **Saliency (gradients)**: higher values mean the prediction changes more if that residue changes. Also low → high, and it does not show positive vs. negative effect.

---

## Extending the App

### Add a new classifier model

1. Define the architecture class in `core/models.py`.
2. Save a trained checkpoint to `checkpoints/`.
3. Add an entry to `MODEL_REGISTRY` in `core/config.py` — the rest of the app picks it up automatically.

### Add a new class label

Update `CLASS_MAP` in `core/config.py`. All prediction tables and explainability logic derive labels from this mapping.
