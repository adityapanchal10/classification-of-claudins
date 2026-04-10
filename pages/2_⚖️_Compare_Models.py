import streamlit as st

from core.config import CLASS_MAP, MODEL_REGISTRY
from core.embeddings import build_baseline_embeddings, get_embedder
from core.explainability import attention_dataframe, compute_ig_attributions, residue_importance_dataframe
from core.io_utils import detect_input_dataframe, validate_sequences
from core.models import load_classifier_bundle
from core.predict import predict_probabilities
from core.ui import DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LENGTH, cache_log, global_sidebar, memory_log, toast_once
from core.visuals import plot_residue_boxplot

st.set_page_config(page_title="Compare Models", layout="wide", page_icon="🧬")
st.logo("🧬")

global_sidebar()


st.title("Compare Models")
seq_length = DEFAULT_SEQ_LENGTH
batch_size = DEFAULT_BATCH_SIZE
ig_steps = st.session_state.get("global_ig_steps", 50)

# Sequence source (mirrors Data Exploration pre-stored flow).
predict_run = st.session_state.get("predict_run")
pre_stored_df = st.session_state.get("input_sequences_df", None)
pre_stored_embeddings = st.session_state.get("generated_embeddings", None)

st.subheader("Sequence Input")
df_valid = None
embeddings_all = None

if pre_stored_df is not None:
    st.info("📌 Using sequences and embeddings from Predict page")
    use_pre_stored = st.checkbox("Use pre-stored data", value=True, key="cmp_use_pre_stored")
    if use_pre_stored:
        df_valid = pre_stored_df.copy()
        embeddings_all = pre_stored_embeddings
    else:
        uploaded = st.file_uploader("Upload FASTA for comparison", type=["fasta", "fa", "faa", "txt"], key="cmp_fasta")
        text_value = st.text_area("Or paste FASTA / one-sequence-per-line text", height=140, key="cmp_text")
        if uploaded is not None or text_value.strip():
            df = validate_sequences(detect_input_dataframe(text_value, uploaded))
            df_valid = df[df["is_valid"]].copy()
else:
    uploaded = st.file_uploader("Upload FASTA for comparison", type=["fasta", "fa", "faa", "txt"], key="cmp_fasta")
    text_value = st.text_area("Or paste FASTA / one-sequence-per-line text", height=140, key="cmp_text")
    if uploaded is not None or text_value.strip():
        df = validate_sequences(detect_input_dataframe(text_value, uploaded))
        df_valid = df[df["is_valid"]].copy()

if df_valid is None:
    st.info("Provide sequences from Predict page or manual input to compare models.")
    st.stop()

if df_valid.empty:
    st.warning("No valid amino acid sequences are available for comparison.")
    st.stop()

if embeddings_all is None:
    with st.spinner("Generating embeddings for comparison..."):
        embedder = get_embedder()
        embedder_name = getattr(embedder, "model_name", "esm_msa1b_t12_100M_UR50S")
        toast_once("_embedder_ready_toast_shown", embedder_name, f"⚗️ Embedder ready: {embedder_name}")
        embeddings_all = embedder.embed_sequences_per_residue(
            df_valid["sequence"].tolist(),
            seq_length=seq_length,
            batch_size=batch_size,
        )
    cache_log("compare cache miss for embeddings; generated fresh embeddings")

if not hasattr(embeddings_all, "shape") or len(embeddings_all.shape) != 3:
    st.error("Expected embeddings shape (num_sequences, num_residues, embedding_dim).")
    st.stop()

sequence_count = min(len(df_valid), embeddings_all.shape[0])
if sequence_count == 0:
    st.warning("No sequences available after aligning with embeddings.")
    st.stop()

df_valid = df_valid.iloc[:sequence_count].reset_index(drop=True)
if hasattr(embeddings_all, "detach"):
    embeddings_all = embeddings_all[:sequence_count]
else:
    embeddings_all = embeddings_all[:sequence_count, :, :]

preselected_idx = 0
if predict_run is not None:
    inspected_result = predict_run.get("inspected_result") if isinstance(predict_run, dict) else None
    if inspected_result is not None and "explain_idx" in inspected_result:
        preselected_idx = int(inspected_result["explain_idx"])
    elif isinstance(predict_run, dict) and "explain_idx" in predict_run:
        preselected_idx = int(predict_run["explain_idx"])
preselected_idx = max(0, min(preselected_idx, sequence_count - 1))

sequence_options = list(range(sequence_count))
selected_idx = st.radio(
    "Select sequence",
    options=sequence_options,
    index=preselected_idx,
    format_func=lambda i: f"{df_valid.iloc[i]['seq_id']} ({df_valid.iloc[i]['length']} aa)",
    key="cmp_selected_sequence_idx",
)

selected_row = df_valid.iloc[selected_idx]
st.caption(f"Selected: {selected_row['seq_id']}")
st.code(selected_row["sequence"], language="text")

st.subheader("Model Selection")
models = list(MODEL_REGISTRY.keys())
default_model = st.session_state.get("global_model_name", models[0])
default_index = models.index(default_model) if default_model in models else 0
col_model_a, col_model_b = st.columns(2)
with col_model_a:
    left_model = st.selectbox("Model A", models, index=default_index, key="cmp_a")
with col_model_b:
    right_model = st.selectbox("Model B", models, index=min(0, len(models)-1), key="cmp_b")

if st.button("Run comparison", type="primary"):
    print(f"[PAGE Compare] Run comparison A={left_model} B={right_model} idx={selected_idx}")
    embedder = get_embedder()
    sample_embedding = embeddings_all[selected_idx].unsqueeze(0)
    baseline_embedding = build_baseline_embeddings(embedder, seq_length)

    cols = st.columns(2)
    for slot, (col, model_name) in enumerate(zip(cols, [left_model, right_model])):
        bundle = load_classifier_bundle(model_name)
        preds, confs, _, attn = predict_probabilities(bundle, sample_embedding)
        residue_attrs, _ = compute_ig_attributions(
            bundle.classifier,
            sample_embedding,
            baseline_embedding,
            int(preds[0]),
            n_steps=ig_steps,
            internal_batch_size=max(4, min(8, ig_steps)),
        )
        trunc_seq = selected_row["sequence"][: sample_embedding.shape[1]]
        ig_df = residue_importance_dataframe(trunc_seq, residue_attrs.squeeze(0).numpy()[: len(trunc_seq)])
        with col:
            st.subheader(model_name)
            st.markdown(f"**Architecture:** {bundle.architecture}")
            st.markdown(f"**Prediction:** {CLASS_MAP[int(preds[0])]} ({confs[0]:.3f})")
            plot_residue_boxplot(ig_df, "score", f"Integrated Gradients — {model_name}", "IG score", key=f"cmp_ig_{slot}")
            if bundle.uses_attention and attn is not None:
                attn_df = attention_dataframe(trunc_seq, attn[0].numpy()[: len(trunc_seq)])
                plot_residue_boxplot(attn_df, "attention", f"Attention Weights — {model_name}", "Attention", key=f"cmp_attn_{slot}")
            else:
                st.info("No attention visualization for this model.")
    print("[PAGE Compare] Comparison done")
    memory_log("compare.run_comparison.done")
