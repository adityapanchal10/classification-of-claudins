import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from core.config import MODEL_REGISTRY, BASE_DIR
from core.embeddings import build_baseline_embeddings, get_embedder, infer_structure_with_esmfold
from core.explainability import (
    attention_dataframe,
    compute_ig_attributions,
    compute_saliency,
    residue_importance_dataframe,
)
from core.io_utils import detect_input_dataframe, validate_sequences
from core.models import load_classifier_bundle
from core.predict import (
    build_prediction_table,
    predict_probabilities,
    resolve_residue_slice,
    slice_embeddings,
    slice_sequence,
    expand_scores_to_full,
)
from core.ui import DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LENGTH, cache_log, global_sidebar, memory_log, toast_once
from core.visuals import plot_attention, plot_importance, plot_top_attributes, show_structure_viewer

st.set_page_config(page_title="Predict", layout="wide", page_icon="🧬")
st.logo("🧬")

global_sidebar()

st.title("Predict")
model_name = st.session_state.get("global_model_name", "Transformer + MLP")
seq_length = DEFAULT_SEQ_LENGTH
batch_size = DEFAULT_BATCH_SIZE
ig_steps = st.session_state.get("global_ig_steps", 50)

cfg = MODEL_REGISTRY[model_name]
default_residue_slice = resolve_residue_slice(cfg)
st.markdown(f"**Model:** {model_name}")
with st.expander("Details", expanded=True):
    st.markdown(f"**Description**: {cfg['description']}")
    st.markdown(f"**Architecture**: {cfg['architecture']}")
    st.markdown(f"**Attention available**: {'Yes' if cfg['uses_attention'] else 'No'}")

REFERENCE_MSA_DIR = BASE_DIR / "reference_msas"

def _resolve_reference_msa(model_name: str, ecs_only: bool):
    """Return the Path to the appropriate reference MSA file, or None."""
    is_diverse = "diverse" in model_name.lower()
    variant = "diverse" if is_diverse else "balanced"
    prefix = "ref_ecs_only_msa" if ecs_only else "ref_full_seqs_msa"
    return REFERENCE_MSA_DIR / f"{prefix}_{variant}.fasta"

# MSA mode is only relevant when the MSA Transformer embedder is selected
_embedder_name = st.session_state.get("global_embedder_name", "MSA Transformer")
_msa_mode_active = (
    _embedder_name == "MSA Transformer"
    and st.session_state.get("global_embed_in_msa_mode", True)
)

ecs_only_default = default_residue_slice is not None
use_ref_msa = st.checkbox(
    "Use Reference MSA for embedding",
    value=False,
    key="predict_use_ref_msa",
    disabled=not _msa_mode_active,
    help="Prepend reference sequences to the input before embedding. Only available when the MSA Transformer embedder is used in MSA mode.",
)
ecs_only = st.checkbox("ECS only (The model will only use the regions specified for the prediction)", value=ecs_only_default, key="predict_ecs_only")
default_ecs1_start, default_ecs1_end, default_ecs2_start, default_ecs2_end = 28, 81, 139, 164
if isinstance(default_residue_slice, list) and len(default_residue_slice) >= 2:
    default_ecs1_start = default_residue_slice[0][0] + 1
    default_ecs1_end = default_residue_slice[0][1]
    default_ecs2_start = default_residue_slice[1][0] + 1
    default_ecs2_end = default_residue_slice[1][1]
cols_ecs = st.columns(4)
with cols_ecs[0]:
    ecs1_start = st.number_input(
        "ECS1 start",
        min_value=1,
        value=int(default_ecs1_start),
        disabled=not ecs_only,
        key="predict_ecs1_start",
    )
with cols_ecs[1]:
    ecs1_end = st.number_input(
        "ECS1 end",
        min_value=1,
        value=int(default_ecs1_end),
        disabled=not ecs_only,
        key="predict_ecs1_end",
    )
with cols_ecs[2]:
    ecs2_start = st.number_input(
        "ECS2 start",
        min_value=1,
        value=int(default_ecs2_start),
        disabled=not ecs_only,
        key="predict_ecs2_start",
    )
with cols_ecs[3]:
    ecs2_end = st.number_input(
        "ECS2 end",
        min_value=1,
        value=int(default_ecs2_end),
        disabled=not ecs_only,
        key="predict_ecs2_end",
    )
if ecs_only:
    ecs1_start_i = int(max(1, ecs1_start))
    ecs1_end_i = int(max(ecs1_start_i, ecs1_end))
    ecs2_start_i = int(max(1, ecs2_start))
    ecs2_end_i = int(max(ecs2_start_i, ecs2_end))
    residue_slice = [(ecs1_start_i - 1, ecs1_end_i), (ecs2_start_i - 1, ecs2_end_i)]
else:
    residue_slice = None

text_value = st.text_area(
    "Enter Sequence(s) here:",
    height=180,
    placeholder=">seq1\nMKT...\n>seq2\nVVV...",
    key="predict_text_input",
)
st.markdown("**OR**")
uploaded_file = st.file_uploader(
    "Upload FASTA",
    type=["fasta", "fa", "faa", "txt"],
    key="predict_upload_file",
)

if st.button("Run inference", type="primary"):
    print(f"[PAGE Predict] Run inference model={model_name}")
    if not text_value.strip() and uploaded_file is None:
        st.warning("Provide sequence input via textbox or file upload.")
        st.stop()
    df = validate_sequences(detect_input_dataframe(text_value, uploaded_file))
    df_valid = df[df["is_valid"]].copy()
    if df_valid.empty:
        st.error("No valid amino acid sequences were found.")
        st.stop()

    with st.spinner(f"Aligning input sequences to in-built reference MSA and generating embeddings..." if use_ref_msa and _msa_mode_active else "Generating embeddings..."):
        embedder = get_embedder()
        embedder_name = getattr(embedder, "model_name", "esm_msa1b_t12_100M_UR50S")
        toast_once("_embedder_ready_toast_shown", embedder_name, f"⚗️ Embedder ready: {embedder_name}")
        msa_only = st.session_state.get("global_embed_in_msa_mode", True)
        if msa_only and getattr(embedder, "supports_msa_mode", True):
            ref_msa_path = _resolve_reference_msa(model_name, ecs_only) if use_ref_msa else None
            embeddings = embedder.embed_msa(
                df_valid["sequence"].tolist(),
                seq_length=seq_length,
                reference_msa_path=ref_msa_path,
            )
        else:
            embeddings = embedder.embed_sequences_per_residue(
                df_valid["sequence"].tolist(),
                seq_length=seq_length,
                batch_size=batch_size,
            )

    bundle = load_classifier_bundle(model_name)
    embeddings_for_model = slice_embeddings(embeddings, residue_slice)
    expected_dim = int(MODEL_REGISTRY[model_name]["kwargs"]["embedding_dim"])
    actual_dim = int(embeddings_for_model.shape[-1])
    if actual_dim != expected_dim:
        st.error(
            f"{model_name} expects {expected_dim}-dim embeddings, but the selected embedder produces {actual_dim}-dim embeddings. "
            "Please recheck the selected classifier matches the embedder."
        )
        st.stop()
    preds, confs, probs, _ = predict_probabilities(bundle, embeddings_for_model, return_attention=False)
    pred_table = build_prediction_table(df_valid, preds, confs, probs)

    print(f"[PAGE Predict] Inference ready n_seq={len(df_valid)}")
    memory_log("predict.run_inference.done")
    st.session_state.input_sequences_df = df_valid.copy()
    cache_log(f"Stored input_sequences_df rows={len(df_valid)}")
    st.session_state.generated_embeddings = embeddings.detach().to(torch.float16) if hasattr(embeddings, "detach") else embeddings
    if hasattr(st.session_state.generated_embeddings, "shape"):
        cache_log(f"Stored predict embeddings shape={tuple(st.session_state.generated_embeddings.shape)}")
    else:
        cache_log("Stored predict embeddings")
    st.session_state.generated_embeddings_embedder = getattr(embedder, "display_name", embedder_name)
    st.session_state.generated_embeddings_msa_only = bool(msa_only and getattr(embedder, "supports_msa_mode", True))
    st.session_state.predict_run = {
        "model_name": model_name,
        "ecs_only": ecs_only,
        "ecs_ranges": (ecs1_start, ecs1_end, ecs2_start, ecs2_end),
        "explain_idx": 0,
        "pred_table": pred_table,
        "inspected_result": None,
    }
    cache_log("Stored predict_run (model, explain_idx, pred_table, inspected_result)")

predict_run = st.session_state.get("predict_run")
shared_df = st.session_state.get("input_sequences_df")
shared_embeddings = st.session_state.get("generated_embeddings")
if (
    predict_run
    and predict_run.get("model_name") == model_name
    and shared_df is not None
    and shared_embeddings is not None
):
    df_valid = shared_df.copy()
    embeddings = shared_embeddings
    pred_table = predict_run.get("pred_table")

    st.subheader("Input dataset")
    if ecs_only and residue_slice is not None:
        df_display = df_valid.copy()
        df_display["ecs_sequence"] = df_display["sequence"].apply(
            lambda seq: slice_sequence(seq, residue_slice)
        )
        st.dataframe(df_display, width='stretch')
    else:
        st.dataframe(df_valid, width='stretch')

    if (
        predict_run.get("ecs_only") != ecs_only
        or tuple(predict_run.get("ecs_ranges") or ()) != (ecs1_start, ecs1_end, ecs2_start, ecs2_end)
    ):
        pred_table = None
        predict_run["pred_table"] = None
        predict_run["inspected_result"] = None
        cache_log("ECS settings changed; cleared cached prediction results")

    if pred_table is None:
        bundle = load_classifier_bundle(model_name)
        embeddings_for_model = slice_embeddings(embeddings, residue_slice)
        preds, confs, probs, _ = predict_probabilities(bundle, embeddings_for_model, return_attention=False)
        pred_table = build_prediction_table(df_valid, preds, confs, probs)
        st.session_state.predict_run["pred_table"] = pred_table
        cache_log("Stored missing predict_run field (pred_table)")

    inspected_result = predict_run.get("inspected_result")

    st.subheader("Predictions")
    st.dataframe(pred_table, width='stretch')
    st.download_button("Download predictions CSV", pred_table.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")

    with st.form("inspect-sequence-form"):
        explain_idx = st.selectbox(
            "**Select sequence to inspect**",
            options=list(range(len(df_valid))),
            index=min(predict_run.get("explain_idx", 0), len(df_valid) - 1),
            format_func=lambda i: f"{df_valid.iloc[i]['description']} ({df_valid.iloc[i]['length']} aa)",
        )
        inspect_clicked = st.form_submit_button("Inspect sequence", type="primary")

    if inspect_clicked:
        print(f"[PAGE Predict] Inspect idx={explain_idx}")
        st.session_state.predict_run["explain_idx"] = explain_idx

        row = df_valid.iloc[explain_idx]
        bundle = load_classifier_bundle(model_name)
        expected_dim = int(MODEL_REGISTRY[model_name]["kwargs"]["embedding_dim"])

        # Reuse the already-computed embeddings instead of re-running the
        # expensive ESM model for a single sequence.
        sample_embedding = embeddings[explain_idx].unsqueeze(0).to(torch.float32)
        sample_embedding = slice_embeddings(sample_embedding, residue_slice)
        if int(sample_embedding.shape[-1]) != expected_dim:
            st.error(
                f"{model_name} expects {expected_dim}-dim embeddings, but the selected embedder produces {int(sample_embedding.shape[-1])}-dim embeddings. "
                "Please recheck the selected classifier matches the embedder."
            )
            st.stop()

        sample_preds, sample_confs, _, sample_attn = predict_probabilities(bundle, sample_embedding)

        baseline_embedding = build_baseline_embeddings(sample_embedding.shape[1], sample_embedding.shape[-1])
        residue_attrs, _ = compute_ig_attributions(
            bundle.classifier,
            sample_embedding,
            baseline_embedding,
            int(sample_preds[0]),
            n_steps=ig_steps,
            internal_batch_size=max(4, min(8, ig_steps)),
        )
        full_seq = row["sequence"][: embeddings.shape[1]]
        trunc_seq = slice_sequence(row["sequence"], residue_slice)
        trunc_seq = trunc_seq[: sample_embedding.shape[1]]
        residue_scores = residue_attrs.squeeze(0).numpy()[: len(trunc_seq)]
        if residue_slice is not None:
            full_scores = expand_scores_to_full(residue_scores, residue_slice, len(full_seq))
            ig_df = residue_importance_dataframe(full_seq, full_scores)
        else:
            ig_df = residue_importance_dataframe(trunc_seq, residue_scores)
        attn_df = None
        saliency_df = None
        if bundle.uses_attention and sample_attn is not None:
            attn_vec = sample_attn[0].numpy()[: len(trunc_seq)]
            if residue_slice is not None:
                full_attn = expand_scores_to_full(attn_vec, residue_slice, len(full_seq))
                attn_df = attention_dataframe(full_seq, full_attn)
            else:
                attn_df = attention_dataframe(trunc_seq, attn_vec)
        elif not bundle.uses_attention:
            _, saliency_attrs = compute_saliency(bundle.classifier, sample_embedding)
            saliency_scores = saliency_attrs.squeeze(0).numpy()[: len(trunc_seq)]
            if residue_slice is not None:
                full_saliency = expand_scores_to_full(saliency_scores, residue_slice, len(full_seq))
                saliency_df = attention_dataframe(full_seq, full_saliency)
            else:
                saliency_df = attention_dataframe(trunc_seq, saliency_scores)
        inspected_result = {
            "explain_idx": explain_idx,
            "seq_id": row["description"],
            "sequence": row["sequence"],
            "trunc_seq": trunc_seq,
            "ig_df": ig_df,
            "attn_df": attn_df,
            "saliency_df": saliency_df,
            "inspect_conf": float(sample_confs[0]),
            "pdb_path": None,
        }
        st.session_state.predict_run["inspected_result"] = inspected_result
        cache_log("Stored predict_run.inspected_result")
        memory_log("predict.inspect_sequence.done")

    if inspected_result is None:
        st.info("Select a sequence and click Inspect sequence to run explainability.")
    else:
        explain_idx = inspected_result["explain_idx"]
        row = df_valid.iloc[explain_idx]
        trunc_seq = inspected_result["trunc_seq"]
        ig_df = inspected_result["ig_df"]
        attn_df = inspected_result.get("attn_df")
        saliency_df = inspected_result.get("saliency_df")
        inspect_conf = inspected_result.get("inspect_conf")
        if inspect_conf is None:
            inspect_conf = float(pred_table.iloc[explain_idx]["confidence"])

        st.markdown(
            f"**Predicted class:** {pred_table.iloc[explain_idx]['predicted_class']}  |  "
            f"**Confidence:** {inspect_conf:.3f}"
        )

        top_pos = ig_df[ig_df["score"] > 0].sort_values("score", ascending=False).head(5).copy()
        top_pos["contribution"] = "Positive"
        top_neg = ig_df[ig_df["score"] < 0].sort_values("score", ascending=True).head(5).copy()
        top_neg["contribution"] = "Negative"
        top_attrs = pd.concat([top_pos, top_neg], ignore_index=True)
        if top_attrs.empty:
            top_attrs = ig_df.reindex(ig_df["score"].abs().sort_values(ascending=False).index).head(10).copy()
            top_attrs["contribution"] = "Neutral"
        plot_top_attributes(top_attrs)

        st.markdown(f"**Residue Importance via Integrated Gradients** - {row['description']}")
        plot_importance(ig_df, "")

        if cfg["uses_attention"] and attn_df is not None:
            st.markdown(f"**Attention Weights** - {row['description']}")
            plot_attention(attn_df, "")
        elif saliency_df is not None:
            st.markdown(f"**Saliency (gradients)** - {row['description']}")
            plot_attention(saliency_df, "", is_saliency=True)
        else:
            st.info("Attention visualization is not available for this model.")

        st.subheader("Structure")
        structure_style = st.radio(
            "Structure style",
            ["cartoon", "sticks", "line", "sphere"],
            index=0,
            horizontal=True,
            key=f"structure_style_{explain_idx}",
        )
        if st.button("Predict structure with ESMFold"):
            structure_sequence = inspected_result.get("sequence", row["sequence"])
            structure_seq_id = inspected_result.get("description", row["description"])
            print(f"[PAGE Predict] ESMFold start seq_id={structure_seq_id}")
            with st.spinner("Running ESMFold..."):
                pdb_path = infer_structure_with_esmfold(structure_sequence, Path(tempfile.gettempdir()) / "protein_sequence_app_v2")
            if pdb_path is None:
                st.error("ESMFold inference is unavailable in this environment. Configure dependencies and retry.")
            else:
                print(f"[PAGE Predict] ESMFold done path={pdb_path}")
                st.session_state.predict_run["inspected_result"]["pdb_path"] = str(pdb_path)
                cache_log(f"Stored predict_run.inspected_result.pdb_path={pdb_path}")
                memory_log("predict.structure.done")

        stored_pdb_path = inspected_result.get("pdb_path")
        if stored_pdb_path:
            pdb_path = Path(stored_pdb_path)
            if pdb_path.exists():
                structure_seq_id = inspected_result.get("description", row["description"])
                show_structure_viewer(pdb_path, residue_importance=ig_df, style_mode=structure_style)
                st.download_button("Download PDB", pdb_path.read_bytes(), file_name=f"{structure_seq_id}.pdb", mime="chemical/x-pdb")
