import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from core.config import MODEL_REGISTRY
from core.embeddings import build_baseline_embeddings, get_embedder, infer_structure_with_esmfold
from core.explainability import attention_dataframe, compute_ig_attributions, residue_importance_dataframe
from core.io_utils import detect_input_dataframe, validate_sequences
from core.models import load_classifier_bundle
from core.predict import build_prediction_table, predict_probabilities
from core.ui import DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LENGTH, cache_log, global_sidebar, memory_log, toast_once
from core.visuals import plot_attention, plot_importance, plot_top_attributes, show_structure_viewer

st.set_page_config(page_title="Predict", layout="wide", page_icon="🧬")
st.logo("🧬")

global_sidebar()

st.title("Predict")
model_name = st.session_state.get("global_model_name", "Transformer + MLP Classifier")
seq_length = DEFAULT_SEQ_LENGTH
batch_size = DEFAULT_BATCH_SIZE
ig_steps = st.session_state.get("global_ig_steps", 50)

cfg = MODEL_REGISTRY[model_name]
st.markdown(f"**Model:** {model_name}")
with st.expander("Details", expanded=True):
    st.markdown(f"**Description**: {cfg['description']}")
    st.markdown(f"**Architecture**: {cfg['architecture']}")
    st.markdown(f"**Attention available**: {'Yes' if cfg['uses_attention'] else 'No'}")

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
    st.subheader("Input dataset")
    st.dataframe(df, width='stretch')
    df_valid = df[df["is_valid"]].copy()
    if df_valid.empty:
        st.error("No valid amino acid sequences were found.")
        st.stop()

    with st.spinner("Generating embeddings..."):
        embedder = get_embedder()
        embedder_name = getattr(embedder, "model_name", "esm_msa1b_t12_100M_UR50S")
        toast_once("_embedder_ready_toast_shown", embedder_name, f"⚗️ Embedder ready: {embedder_name}")
        embeddings = embedder.embed_msa(
            df_valid["sequence"].tolist(),
            seq_length=seq_length,
        )

    bundle = load_classifier_bundle(model_name)
    preds, confs, probs, _ = predict_probabilities(bundle, embeddings, return_attention=False)
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
    st.session_state.predict_run = {
        "model_name": model_name,
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

    if pred_table is None:
        bundle = load_classifier_bundle(model_name)
        preds, confs, probs, _ = predict_probabilities(bundle, embeddings, return_attention=False)
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
            format_func=lambda i: f"{df_valid.iloc[i]['seq_id']} ({df_valid.iloc[i]['length']} aa)",
        )
        inspect_clicked = st.form_submit_button("Inspect sequence", type="primary")

    if inspect_clicked:
        print(f"[PAGE Predict] Inspect idx={explain_idx}")
        st.session_state.predict_run["explain_idx"] = explain_idx

        row = df_valid.iloc[explain_idx]
        bundle = load_classifier_bundle(model_name)

        # Reuse the already-computed embeddings instead of re-running the
        # expensive ESM model for a single sequence.
        sample_embedding = embeddings[explain_idx].unsqueeze(0).to(torch.float32)

        sample_preds, sample_confs, _, sample_attn = predict_probabilities(bundle, sample_embedding)

        baseline_embedding = build_baseline_embeddings(seq_length)
        residue_attrs, _ = compute_ig_attributions(
            bundle.classifier,
            sample_embedding,
            baseline_embedding,
            int(sample_preds[0]),
            n_steps=ig_steps,
            internal_batch_size=max(4, min(8, ig_steps)),
        )
        trunc_seq = row["sequence"][: sample_embedding.shape[1]]
        ig_df = residue_importance_dataframe(trunc_seq, residue_attrs.squeeze(0).numpy()[: len(trunc_seq)])
        attn_df = None
        if bundle.uses_attention and sample_attn is not None:
            attn_vec = sample_attn[0].numpy()[: len(trunc_seq)]
            attn_df = attention_dataframe(trunc_seq, attn_vec)
        inspected_result = {
            "explain_idx": explain_idx,
            "seq_id": row["seq_id"],
            "sequence": row["sequence"],
            "trunc_seq": trunc_seq,
            "ig_df": ig_df,
            "attn_df": attn_df,
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

        st.markdown(f"**Residue Importance via Integrated Gradients** - {row['seq_id']}")
        plot_importance(ig_df, "")

        if cfg["uses_attention"] and attn_df is not None:
            st.markdown(f"**Attention Weights** - {row['seq_id']}")
            plot_attention(attn_df, "")
        else:
            st.info("Attention visualization is not available for this model.")

        st.subheader("Structure")
        structure_style = st.radio(
            "Structure style",
            ["sticks", "cartoon", "line", "sphere"],
            index=0,
            horizontal=True,
            key=f"structure_style_{explain_idx}",
        )
        if st.button("Predict structure with ESMFold"):
            structure_sequence = inspected_result.get("sequence", row["sequence"])
            structure_seq_id = inspected_result.get("seq_id", row["seq_id"])
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
                structure_seq_id = inspected_result.get("seq_id", row["seq_id"])
                show_structure_viewer(pdb_path, residue_importance=ig_df, style_mode=structure_style)
                st.download_button("Download PDB", pdb_path.read_bytes(), file_name=f"{structure_seq_id}.pdb", mime="chemical/x-pdb")
