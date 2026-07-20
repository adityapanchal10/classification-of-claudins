# Functional Classification of Claudins - Streamlit App

A multi-page Streamlit application for classifying claudin sequences using ESM embeddings from the MSA Transformer (`esm_msa1b_t12_100M_UR50S`) and ESM2 (`esm2_t30_150M_UR50D`), together with a family of trained classifiers. The app supports batch inference (with reference MSA), per-residue explainability, side-by-side model comparison, embedding distribution exploration, and reference MSA browsing.

---

## Pages

| Page | Purpose |
|---|---|
| 🔮 **Predict** | Run inference on one or more sequences. Inspect any sequence with Integrated Gradients (IG), saliency, and attention heatmaps (for models using attention). Optionally align input against a reference MSA before embedding. Predict structure through the public ESMFold API and color it by residue importance. Results persist in session state for use in other pages. |
| ⚖️ **Compare Models** | Select two models and compare their predictions and explainability side-by-side for any sequence already run on the Predict page (or new input). Each column has its own embedder and MSA-mode selector. Uses normalised per-residue bar charts for IG scores and attention weights. |
| 📊 **Data Exploration** | Visualise per-residue embedding distributions using PCA. Uses pre-stored embeddings when available, or generates embeddings from the provided input sequences. Filter sequences via a multiselect widget with sticky frosted-glass controls and run exploration on demand. |
| ℹ️ **About Models** | Overview table of all registered models. Shows a `torchinfo` architecture summary and training checkpoint metrics (saved epoch, validation AUC, accuracy, loss, and % class error curves) for the selected model. |
| 📖 **Reference MSAs** | Browse the four reference MSAs used for alignment-aware embedding. Displays per-MSA sequence counts broken down by claudin family and functional class. |

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

| Model | Architecture summary | Embedder | Attention |
|---|---|---|---|
| **Transformer + MLP** | Linear projection → positional embedding → self-attention blocks → attention/mean/max pooling → fusion MLP → linear head | MSA Transformer | ✅ |
| **Transformer + MLP (ECS only)** | As above, trained on ECS1/2 residue slices | MSA Transformer | ✅ |
| **Transformer + MLP Non-MSA** | As above, trained without MSA context | MSA Transformer | ✅ |
| **Transformer + MLP Non-MSA (ECS only)** | As above, Non-MSA + ECS-only | MSA Transformer | ✅ |
| **Transformer + MLP ESM2** | As above, trained on ESM2 (640-d) embeddings | ESM2 | ✅ |
| **Transformer + MLP ESM2 (ECS only)** | As above, ESM2 + ECS-only | ESM2 | ✅ |
| **Simple Linear** | LayerNorm → learned attention scores → softmax-weighted sum → dropout → linear head | MSA Transformer | ❌ |
| **Simple Linear (ECS only)** | As above, ECS-only | MSA Transformer | ❌ |
| **Simple Linear Non-MSA** | As above, trained without MSA context | MSA Transformer | ❌ |
| **Simple Linear Non-MSA (ECS only)** | As above, Non-MSA + ECS-only | MSA Transformer | ❌ |
| **Simple Linear ESM2** | As above, trained on ESM2 (640-d) embeddings | ESM2 | ❌ |
| **Simple Linear ESM2 (ECS only)** | As above, ESM2 + ECS-only | ESM2 | ❌ |
| **Simple Linear Diverse** | As Simple Linear, trained with diversity-aware chunked batches (round-robin, ≥1 sequence per claudin family per batch) | MSA Transformer | ❌ |
| **Simple Linear Diverse (ECS only)** | As above, ECS-only | MSA Transformer | ❌ |
| **Simple Linear Balanced** | As Simple Linear, trained with balanced chunked batches (equal sequences per claudin family per batch) | MSA Transformer | ❌ |
| **Simple Linear Balanced (ECS only)** | As above, ECS-only | MSA Transformer | ❌ |
| **Simple Linear Family** | As Simple Linear, trained with family-grouped chunked batches | MSA Transformer | ❌ |
| **Simple Linear Family (ECS only)** | As above, ECS-only | MSA Transformer | ❌ |
| **Transformer + MLP Diverse** | As Transformer + MLP, trained with diversity-aware chunked batches (round-robin, ≥1 sequence per claudin family per batch) | MSA Transformer | ✅ |
| **Transformer + MLP Diverse (ECS only)** | As above, ECS-only | MSA Transformer | ✅ |
| **Transformer + MLP Balanced** | As Transformer + MLP, trained with balanced chunked batches (equal sequences per claudin family per batch) | MSA Transformer | ✅ |
| **Transformer + MLP Balanced (ECS only)** | As above, ECS-only | MSA Transformer | ✅ |
| **Transformer + MLP Family** | As Transformer + MLP, trained with family-grouped chunked batches | MSA Transformer | ✅ |
| **Transformer + MLP Family (ECS only)** | As above, ECS-only | MSA Transformer | ✅ |
| **Mamba2** | Input projection → selective SSM block → LayerNorm → attention pooling → classifier head | ESM2 | ✅ |
| **Mamba2 (ECS only)** | As above, ECS-only | ESM2 | ✅ |

Checkpoints live in `checkpoints/`. Each `.pt` file stores the model weights. Checkpoints not present locally are fetched automatically from Google Drive on first use.

**Compatibility notes:**
- Models suffixed with `ESM2` were trained on ESM2 (640-d) embeddings - select the `ESM2` embedder for these. The sidebar and Compare page filter model lists by the selected embedder to prevent mismatches.
- Models marked `Non-MSA` were trained with the MSA Transformer in non-MSA (single-sequence) mode — keep the `MSA Transformer` embedder selected and turn the MSA mode toggle **off**.
- `ECS only` variants were trained on the ECS1/ECS2 residue slices (`residue_slice`: positions 27–81 and 138–164, 0-based).
- `Diverse` variants used diversity-aware batching (round-robin per claudin family). `Balanced` variants used equal-per-family batching. `Family` variants used family-grouped batching.

---

## Project Structure

```
Home.py                         # Landing page, sidebar state initialisation
requirements.txt
packages.txt
checkpoints/                    # Model checkpoint .pt files and ESM alphabet
reference_msas/                 # Four reference MSA FASTA files (balanced/diverse × full/ECS)
pages/
    1_🔮_Predict.py             # Inference, explainability, reference MSA embedding, structure prediction
    2_⚖️_Compare_Models.py      # Side-by-side model comparison (per-column embedder selectors)
    3_📊_Data_Exploration.py    # PCA embedding visualisation
    4_ℹ️_About_Models.py        # Model registry overview and checkpoint stats
    5_📖_Reference_MSAs.py      # Reference MSA browser with family/class breakdown
core/
    config.py                   # CLASS_MAP, MODEL_REGISTRY, CHECKPOINT_GDRIVE_URLS, path constants
    models.py                   # Classifier architectures and checkpoint loading
    io_utils.py                 # FASTA / plain-text parsing and sequence validation
    embeddings.py               # ESM-MSA-1b / ESM2 per-residue embedder, ESMFold API helper
    predict.py                  # predict_probabilities(), build_prediction_table(), slice helpers
    explainability.py           # Integrated Gradients, saliency, attention and IG dataframes
    visuals.py                  # Plotly charts: heatmaps, bar charts, PCA plots, structure viewer
    ui.py                       # global_sidebar(), app_header(), shared defaults, cache/
```

---

## Session State Keys

Pages share data through `st.session_state`:

| Key | Set by | Used by |
|---|---|---|
| `input_sequences_df` | Predict | Compare Models, Data Exploration |
| `generated_embeddings` | Predict | Compare Models, Data Exploration |
| `generated_embeddings_embedder` | Predict | Tracks which embedder produced `generated_embeddings` |
| `generated_embeddings_msa_only` | Predict | Tracks whether `generated_embeddings` were produced in MSA mode |
| `predict_run` | Predict | Compare Models (pre-selects inspected sequence) |
| `global_model_name` | Sidebar / any page | All pages |
| `global_embedder_name` | Sidebar | All pages |
| `global_embed_in_msa_mode` | Sidebar | Predict, Compare Models |
| `global_ig_steps` | Sidebar | Predict, Compare Models |
| `global_enable_memory_logs` | Sidebar | All pages (optional RSS memory logging) |

---

## Visualisations

| Chart | Where | Details |
|---|---|---|
| Residue heatmap (IG / saliency / attention) | Predict | Fixed 13 px cells, horizontal scroll, drag pan, double-click 1.5× zoom, fixed transparent colorbar |
| Per-residue bar chart (IG / attention) | Compare Models | Normalised to [−1, 1] or [0, 1]; diverging RdBu for IG, Blues for attention; theme-aware |
| PCA residue boxplots + heatmap | Data Exploration | One box per sequence per residue; explained-variance table; theme-aware diverging heatmap |
| Sequence summary scatter | Data Exploration | Mean norm vs. mean spread across sequences |
| 3-D structure viewer | Predict | ESMFold API structure fetch; py3Dmol rendering colored by residue contribution to the prediction; PDB download |

---

## Running the App

Try out the app here: https://classification-of-claudins.streamlit.app/

**OR**

run locally:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run Home.py
```

The ESM MSA-1b and ESM2 model weights are downloaded automatically on first run via the `fair-esm` library.
Classifier checkpoints not present in `checkpoints/` are fetched from Google Drive on first use via `gdown`. Once fetched, all weights are cached locally and reused.
ESMFold structure prediction is fetched from the public API at `https://api.esmatlas.com/foldSequence/v1/pdb/` and is optional. Residue importance from IG is written into the structure viewer's B-factors for coloring.

---

## Reference MSA Embedding

The Predict and Compare pages expose a **Use Reference MSA for embedding** checkbox (available when the MSA Transformer embedder is active in MSA mode). When enabled, input sequences are aligned against one of four pre-built reference MSAs before embedding:

| Variant | Description |
|---|---|
| **Full sequences — Balanced** | Full claudin sequences, equal sequences per family |
| **Full sequences — Diverse** | Full claudin sequences, training-like variety |
| **ECS only — Balanced** | ECS1/2 segments only, equal sequences per family |
| **ECS only — Diverse** | ECS1/2 segments only, training-like variety |

The appropriate default variant is pre-selected based on the active model name (e.g. `Diverse` models default to the diverse MSA).

---

## ECS-Only Mode

Predict and Compare pages include an **ECS only** toggle. When enabled, provide ECS1 and ECS2 ranges (1-based, inclusive). The app **snips the input sequences to the specified ECS regions first**, then passes the shorter sequences to the embedder. This means the ESM model only sees ECS residues, which avoids any influence from non-ECS context making it consistent to setup in training. After inference, explainability scores (IG, saliency, attention) are expanded back onto the full sequence for display, with non-ECS positions set to zero. Use models with `ECS only` in their name for best results when this toggle is on.

---

## MSA Mode Toggle

The sidebar includes an **Embed in MSA mode** toggle (active only for the MSA Transformer embedder). When enabled, embeddings are generated with full MSA context; when disabled, sequences are embedded independently. Use `Non-MSA` models when the toggle is off. The toggle is disabled automatically when ESM2 is selected.

In Compare Models, each column has its own MSA toggle, allowing direct MSA-on vs. MSA-off comparisons side by side.

---

## Embedder Options and Compatibility

| Embedder | Embedding dim | MSA mode | Token handling |
|---|---|---|---|
| **MSA Transformer** (`esm_msa1b_t12_100M_UR50S`) | 768 | ✅ supported | Leading BOS token removed |
| **ESM2** (`esm2_t30_150M_UR50D`) | 640 | ❌ not supported | Both BOS and EOS tokens removed |

The sidebar and Compare page model dropdowns are filtered to only show models compatible with the currently selected embedder. Pre-stored embeddings from the Predict page are reused on Compare only when embedder name and MSA-mode both match.

---

## Interpreting Explainability

- **Integrated Gradients (IG)**: measures how much each residue contributed to the predicted class. Supports positive and negative attribution (diverging RdBu colorscale).
- **Saliency**: gradient magnitude at the input - how sensitive the prediction is to perturbations at each residue. Always non-negative (no directional information).
- **Attention**: the model's internal attention weights. Higher values indicate residues the model focuses on more. Always non-negative.

---

## Extending the App

### Add a new classifier model

1. Define the architecture class in `core/models.py`.
2. Save a trained checkpoint to `checkpoints/` (or add a Google Drive URL to `CHECKPOINT_GDRIVE_URLS` in `core/config.py`).
3. Add an entry to `MODEL_REGISTRY` in `core/config.py` — the rest of the app picks it up automatically.

### Add a new class label

Update `CLASS_MAP` in `core/config.py`. All prediction tables and explainability logic derive labels from this mapping.

### Override checkpoint URLs at runtime

Set the `CHECKPOINT_GDRIVE_URLS_JSON` environment variable to a JSON object mapping model names to Drive URLs. These are merged on top of the defaults in `config.py`.

---

## References

- MSA Transformer:
```bibtex
@article{rao2021msa,
  author = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
  title={MSA Transformer},
  year={2021},
  doi={10.1101/2021.02.12.430858},
  url={https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1},
  journal={bioRxiv}
}
```

- ESM2 and ESMFold:
```bibtex
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
