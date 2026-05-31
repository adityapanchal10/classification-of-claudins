import streamlit as st
import torch

from core.config import CLASS_MAP, MODEL_REGISTRY
from core.embeddings import (
    DEFAULT_EMBEDDER_NAME,
    available_embedder_names,
    build_baseline_embeddings,
    embedder_supports_msa_mode,
    get_embedder,
)
from core.explainability import (
    attention_dataframe,
    compute_ig_attributions,
    compute_saliency,
    residue_importance_dataframe,
)
from core.io_utils import detect_input_dataframe, validate_sequences
from core.models import load_classifier_bundle
from core.predict import (
    predict_probabilities,
    resolve_residue_slice,
    slice_embeddings,
    slice_sequence,
    expand_scores_to_full,
)
from core.ui import DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LENGTH, cache_log, memory_log, toast_once
from core.visuals import plot_residue_boxplot

st.set_page_config(page_title="Compare Models", layout="wide", page_icon="🧬")
st.logo("🧬")


st.title("Compare Models")
seq_length = DEFAULT_SEQ_LENGTH
batch_size = DEFAULT_BATCH_SIZE
ig_steps = st.session_state.get("global_ig_steps", 50)
global_msa_default = st.session_state.get("global_embed_in_msa_mode", True)
default_embedder_name = st.session_state.get("global_embedder_name", DEFAULT_EMBEDDER_NAME)
embedder_options = available_embedder_names()
if default_embedder_name not in embedder_options:
    default_embedder_name = DEFAULT_EMBEDDER_NAME

# Sequence source (mirrors Data Exploration pre-stored flow).
predict_run = st.session_state.get("predict_run")
pre_stored_df = st.session_state.get("input_sequences_df", None)
pre_stored_embeddings = st.session_state.get("generated_embeddings", None)
pre_stored_embeddings_embedder = st.session_state.get("generated_embeddings_embedder")
pre_stored_embeddings_msa_only = st.session_state.get("generated_embeddings_msa_only")

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

if embeddings_all is not None and (not hasattr(embeddings_all, "shape") or len(embeddings_all.shape) != 3):
    st.error("Expected embeddings shape (num_sequences, num_residues, embedding_dim).")
    st.stop()

sequence_count = len(df_valid) if embeddings_all is None else min(len(df_valid), embeddings_all.shape[0])
if sequence_count == 0:
    st.warning("No sequences available after aligning with embeddings.")
    st.stop()

df_valid = df_valid.iloc[:sequence_count].reset_index(drop=True)
if embeddings_all is not None:
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
    format_func=lambda i: f"{df_valid.iloc[i]['description']} ({df_valid.iloc[i]['length']} aa)",
    key="cmp_selected_sequence_idx",
)

selected_row = df_valid.iloc[selected_idx]
st.caption(f"Selected: {selected_row['description']}")
st.code(selected_row["sequence"], language="text")

st.subheader("Model Selection")
all_models = list(MODEL_REGISTRY.keys())
default_model = st.session_state.get("global_model_name", all_models[0])
col_model_a, col_model_b = st.columns(2)
with col_model_a:
    if st.session_state.get("cmp_a_embedder") not in embedder_options:
        st.session_state["cmp_a_embedder"] = default_embedder_name
    left_embedder = st.selectbox("Embedder A", embedder_options, key="cmp_a_embedder")
    left_msa_supported = embedder_supports_msa_mode(left_embedder)
    if not left_msa_supported:
        st.session_state["cmp_a_msa_only"] = False
    elif "cmp_a_msa_only" not in st.session_state:
        st.session_state["cmp_a_msa_only"] = global_msa_default
    left_msa_only = st.toggle(
        "Embed in MSA mode",
        value=st.session_state["cmp_a_msa_only"],
        key="cmp_a_msa_only",
        disabled=not left_msa_supported,
    )
    # Show only models compatible with the selected embedder for column A
    left_models = []
    for mn, meta in MODEL_REGISTRY.items():
        compat = meta.get("compatible_embedder")
        if isinstance(compat, (list, tuple)):
            if left_embedder in compat:
                left_models.append(mn)
        else:
            if compat == left_embedder:
                left_models.append(mn)
    if not left_models:
        left_models = all_models
        st.info(f"No models found compatible with {left_embedder}; showing all models for Model A.")
    if st.session_state.get("cmp_a") not in left_models:
        st.session_state["cmp_a"] = default_model if default_model in left_models else left_models[0]
    left_model = st.selectbox(
        "Model A",
        left_models,
        index=left_models.index(st.session_state["cmp_a"]) if st.session_state.get("cmp_a") in left_models else 0,
        key="cmp_a",
    )
    left_defaults = resolve_residue_slice(MODEL_REGISTRY[left_model])
    left_ecs_default = left_defaults is not None
    left_ecs_only = st.checkbox("ECS only", value=left_ecs_default, key="cmp_a_ecs_only")
    default_ecs1_start, default_ecs1_end, default_ecs2_start, default_ecs2_end = 28, 81, 139, 164
    if isinstance(left_defaults, list) and len(left_defaults) >= 2:
        default_ecs1_start = left_defaults[0][0] + 1
        default_ecs1_end = left_defaults[0][1]
        default_ecs2_start = left_defaults[1][0] + 1
        default_ecs2_end = left_defaults[1][1]
    left_cols = st.columns(4)
    with left_cols[0]:
        left_ecs1_start = st.number_input(
            "ECS1 start",
            min_value=1,
            value=int(default_ecs1_start),
            disabled=not left_ecs_only,
            key="cmp_a_ecs1_start",
        )
    with left_cols[1]:
        left_ecs1_end = st.number_input(
            "ECS1 end",
            min_value=1,
            value=int(default_ecs1_end),
            disabled=not left_ecs_only,
            key="cmp_a_ecs1_end",
        )
    with left_cols[2]:
        left_ecs2_start = st.number_input(
            "ECS2 start",
            min_value=1,
            value=int(default_ecs2_start),
            disabled=not left_ecs_only,
            key="cmp_a_ecs2_start",
        )
    with left_cols[3]:
        left_ecs2_end = st.number_input(
            "ECS2 end",
            min_value=1,
            value=int(default_ecs2_end),
            disabled=not left_ecs_only,
            key="cmp_a_ecs2_end",
        )
with col_model_b:
    if st.session_state.get("cmp_b_embedder") not in embedder_options:
        st.session_state["cmp_b_embedder"] = default_embedder_name
    right_embedder = st.selectbox("Embedder B", embedder_options, key="cmp_b_embedder")
    right_msa_supported = embedder_supports_msa_mode(right_embedder)
    if not right_msa_supported:
        st.session_state["cmp_b_msa_only"] = False
    elif "cmp_b_msa_only" not in st.session_state:
        st.session_state["cmp_b_msa_only"] = global_msa_default
    right_msa_only = st.toggle(
        "Embed in MSA mode",
        value=st.session_state["cmp_b_msa_only"],
        key="cmp_b_msa_only",
        disabled=not right_msa_supported,
    )
    # Show only models compatible with the selected embedder for column B
    right_models = []
    for mn, meta in MODEL_REGISTRY.items():
        compat = meta.get("compatible_embedder")
        if isinstance(compat, (list, tuple)):
            if right_embedder in compat:
                right_models.append(mn)
        else:
            if compat == right_embedder:
                right_models.append(mn)
    if not right_models:
        right_models = all_models
        st.info(f"No models found compatible with {right_embedder}; showing all models for Model B.")
    if st.session_state.get("cmp_b") not in right_models:
        st.session_state["cmp_b"] = default_model if default_model in right_models else right_models[0]
    right_model = st.selectbox(
        "Model B",
        right_models,
        index=right_models.index(st.session_state["cmp_b"]) if st.session_state.get("cmp_b") in right_models else 0,
        key="cmp_b",
    )
    right_defaults = resolve_residue_slice(MODEL_REGISTRY[right_model])
    right_ecs_default = right_defaults is not None
    right_ecs_only = st.checkbox("ECS only", value=right_ecs_default, key="cmp_b_ecs_only")
    default_ecs1_start, default_ecs1_end, default_ecs2_start, default_ecs2_end = 28, 81, 139, 164
    if isinstance(right_defaults, list) and len(right_defaults) >= 2:
        default_ecs1_start = right_defaults[0][0] + 1
        default_ecs1_end = right_defaults[0][1]
        default_ecs2_start = right_defaults[1][0] + 1
        default_ecs2_end = right_defaults[1][1]
    right_cols = st.columns(4)
    with right_cols[0]:
        right_ecs1_start = st.number_input(
            "ECS1 start",
            min_value=1,
            value=int(default_ecs1_start),
            disabled=not right_ecs_only,
            key="cmp_b_ecs1_start",
        )
    with right_cols[1]:
        right_ecs1_end = st.number_input(
            "ECS1 end",
            min_value=1,
            value=int(default_ecs1_end),
            disabled=not right_ecs_only,
            key="cmp_b_ecs1_end",
        )
    with right_cols[2]:
        right_ecs2_start = st.number_input(
            "ECS2 start",
            min_value=1,
            value=int(default_ecs2_start),
            disabled=not right_ecs_only,
            key="cmp_b_ecs2_start",
        )
    with right_cols[3]:
        right_ecs2_end = st.number_input(
            "ECS2 end",
            min_value=1,
            value=int(default_ecs2_end),
            disabled=not right_ecs_only,
            key="cmp_b_ecs2_end",
        )

if st.button("Run comparison", type="primary"):
    print(f"[PAGE Compare] Run comparison A={left_model} B={right_model} idx={selected_idx}")
    cols = st.columns(2)
    embedder_ref = {}
    embeddings_cache = {}

    def _get_embeddings_for_model(embedder_name: str, msa_only: bool):
        effective_msa_only = bool(msa_only and embedder_supports_msa_mode(embedder_name))
        cache_key = (embedder_name, effective_msa_only)
        if cache_key in embeddings_cache:
            return embeddings_cache[cache_key]

        if (
            embeddings_all is not None
            and pre_stored_embeddings_embedder == embedder_name
            and pre_stored_embeddings_msa_only == effective_msa_only
        ):
            if effective_msa_only:
                embeddings_cache[cache_key] = embeddings_all
            else:
                embeddings_cache[cache_key] = embeddings_all[selected_idx:selected_idx + 1]
            return embeddings_cache[cache_key]

        if embedder_name not in embedder_ref:
            embedder_ref[embedder_name] = get_embedder(embedder_name)
            embedder_name_resolved = getattr(embedder_ref[embedder_name], "model_name", embedder_name)
            toast_once("_embedder_ready_toast_shown", embedder_name_resolved, f"⚗️ Embedder ready: {embedder_name_resolved}")

        embedder = embedder_ref[embedder_name]
        if effective_msa_only:
            embeddings_cache[cache_key] = embedder.embed_msa(
                df_valid["sequence"].tolist(),
                seq_length=seq_length,
            )
        else:
            embeddings_cache[cache_key] = embedder.embed_sequences_per_residue(
                [selected_row["sequence"]],
                seq_length=seq_length,
                batch_size=batch_size,
            )
        return embeddings_cache[cache_key]

    def _get_sample_embedding(embedder_name: str, msa_only: bool):
        embeddings_for_mode = _get_embeddings_for_model(embedder_name, msa_only)
        effective_msa_only = bool(msa_only and embedder_supports_msa_mode(embedder_name))
        if effective_msa_only:
            return embeddings_for_mode[selected_idx].unsqueeze(0)
        return embeddings_for_mode[0].unsqueeze(0)

    for slot, (col, model_name, embedder_name, msa_only_setting) in enumerate(
        zip(cols, [left_model, right_model], [left_embedder, right_embedder], [left_msa_only, right_msa_only])
    ):
        bundle = load_classifier_bundle(model_name)
        expected_dim = int(MODEL_REGISTRY[model_name]["kwargs"]["embedding_dim"])
        if slot == 0:
            msa_only = msa_only_setting
            if left_ecs_only:
                ecs1_start = int(max(1, left_ecs1_start))
                ecs1_end = int(max(ecs1_start, left_ecs1_end))
                ecs2_start = int(max(1, left_ecs2_start))
                ecs2_end = int(max(ecs2_start, left_ecs2_end))
                residue_slice = [(ecs1_start - 1, ecs1_end), (ecs2_start - 1, ecs2_end)]
            else:
                residue_slice = None
        else:
            msa_only = msa_only_setting
            if right_ecs_only:
                ecs1_start = int(max(1, right_ecs1_start))
                ecs1_end = int(max(ecs1_start, right_ecs1_end))
                ecs2_start = int(max(1, right_ecs2_start))
                ecs2_end = int(max(ecs2_start, right_ecs2_end))
                residue_slice = [(ecs1_start - 1, ecs1_end), (ecs2_start - 1, ecs2_end)]
            else:
                residue_slice = None
        sample_embedding = _get_sample_embedding(embedder_name, msa_only).to(torch.float32)
        sample_embedding = slice_embeddings(sample_embedding, residue_slice)
        if int(sample_embedding.shape[-1]) != expected_dim:
            with col:
                st.error(
                    f"{model_name} expects {expected_dim}-dim embeddings, but {embedder_name} produces {int(sample_embedding.shape[-1])}-dim embeddings. "
                    "Please recheck the selected classifier matches the embedder."
                )
            continue
        baseline_embedding = build_baseline_embeddings(sample_embedding.shape[1], sample_embedding.shape[-1])
        preds, confs, _, attn = predict_probabilities(bundle, sample_embedding)
        residue_attrs, _ = compute_ig_attributions(
            bundle.classifier,
            sample_embedding,
            baseline_embedding,
            int(preds[0]),
            n_steps=ig_steps,
            internal_batch_size=max(4, min(8, ig_steps)),
        )
        full_seq = selected_row["sequence"][: sample_embedding.shape[1]]
        trunc_seq = slice_sequence(selected_row["sequence"], residue_slice)
        trunc_seq = trunc_seq[: sample_embedding.shape[1]]
        residue_scores = residue_attrs.squeeze(0).numpy()[: len(trunc_seq)]
        if residue_slice is not None:
            full_scores = expand_scores_to_full(residue_scores, residue_slice, len(full_seq))
            ig_df = residue_importance_dataframe(full_seq, full_scores)
        else:
            ig_df = residue_importance_dataframe(trunc_seq, residue_scores)
        with col:
            st.subheader(model_name)
            st.markdown(f"**Architecture:** {bundle.architecture}")
            st.markdown(f"**Prediction:** {CLASS_MAP[int(preds[0])]} ({confs[0]:.3f})")
            plot_residue_boxplot(ig_df, "score", f"Integrated Gradients — {model_name}", "IG score", key=f"cmp_ig_{slot}")
            if bundle.uses_attention and attn is not None:
                attn_vec = attn[0].numpy()[: len(trunc_seq)]
                if residue_slice is not None:
                    full_attn = expand_scores_to_full(attn_vec, residue_slice, len(full_seq))
                    attn_df = attention_dataframe(full_seq, full_attn)
                else:
                    attn_df = attention_dataframe(trunc_seq, attn_vec)
                plot_residue_boxplot(attn_df, "attention", f"Attention Weights — {model_name}", "Attention", key=f"cmp_attn_{slot}")
            else:
                # st.info("No attention visualization for this model. But we can compute saliency!")
                _, saliency_attrs = compute_saliency(bundle.classifier, sample_embedding)
                saliency_scores = saliency_attrs.squeeze(0).numpy()[: len(trunc_seq)]
                if residue_slice is not None:
                    full_saliency = expand_scores_to_full(saliency_scores, residue_slice, len(full_seq))
                    saliency_df = attention_dataframe(full_seq, full_saliency)
                else:
                    saliency_df = attention_dataframe(trunc_seq, saliency_scores)
                plot_residue_boxplot(saliency_df, "attention", f"Saliency — {model_name}", "Saliency", key=f"cmp_sal_{slot}")
    print("[PAGE Compare] Comparison done")
    memory_log("compare.run_comparison.done")
