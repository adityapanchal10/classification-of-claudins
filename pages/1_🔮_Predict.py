import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core.embeddings import build_baseline_embeddings, get_embedder, infer_structure_with_esmfold
from core.explainability import attention_dataframe, compute_ig_attributions, residue_importance_dataframe
from core.io_utils import detect_input_dataframe, validate_sequences
from core.models import load_classifier_bundle
from core.predict import build_prediction_table, predict_probabilities
from core.ui import DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LENGTH, global_sidebar
from core.visuals import plot_attention, plot_importance, plot_top_attributes, show_structure_viewer

global_sidebar()

st.title("Predict")
model_name = st.session_state.get("global_model_name", "Transformer + MLP Classifier")
seq_length = DEFAULT_SEQ_LENGTH
batch_size = DEFAULT_BATCH_SIZE
ig_steps = st.session_state.get("global_ig_steps", 50)

bundle = load_classifier_bundle(model_name)
st.markdown(f"**Model:** {model_name}")
with st.expander("Details", expanded=True):
    st.markdown(f"**Description**: {bundle.description}")
    st.markdown(f"**Architecture**: {bundle.architecture}")
    st.markdown(f"**Attention available**: {'Yes' if bundle.uses_attention else 'No'}")

text_value = st.text_area("Enter Sequence(s) here:", height=180, placeholder=">seq1\nMKT...\n>seq2\nVVV...")
st.markdown("**OR**")
uploaded_file = st.file_uploader("Upload FASTA", type=["fasta", "fa", "faa", "txt"])

if st.button("Run inference", type="primary"):
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

    with st.spinner("Generating embeddings, Downloading model..."):
        embedder = get_embedder()
        embeddings = embedder.embed_sequences_per_residue(df_valid["sequence"].tolist(), seq_length=seq_length, batch_size=batch_size)

    preds, confs, probs, attn = predict_probabilities(bundle, embeddings)
    pred_table = build_prediction_table(df_valid, preds, confs, probs)
    st.session_state.input_sequences_df = df_valid.copy()
    st.session_state.generated_embeddings = embeddings  # Store reference, no need to clone
    st.session_state.predict_run = {
        "model_name": model_name,
        "df_valid": df_valid.copy(),
        "embeddings": embeddings,  # Store reference, not a copy
        "preds": preds,
        "confs": confs,
        "attn": attn,
        "pred_table": pred_table.copy(),
        "explain_idx": 0,
        "inspected_result": None,
    }

predict_run = st.session_state.get("predict_run")
if predict_run and predict_run.get("model_name") == model_name:
    df_valid = predict_run["df_valid"]
    embeddings = predict_run["embeddings"]
    preds = predict_run["preds"]
    confs = predict_run["confs"]
    attn = predict_run["attn"]
    pred_table = predict_run["pred_table"]

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
        st.session_state.predict_run["explain_idx"] = explain_idx

        row = df_valid.iloc[explain_idx]
        sample_embedding = embeddings[explain_idx].unsqueeze(0)
        sample_pred = int(preds[explain_idx])
        baseline_embedding = build_baseline_embeddings(get_embedder(), seq_length)
        residue_attrs, delta = compute_ig_attributions(
            bundle.classifier,
            sample_embedding,
            baseline_embedding,
            sample_pred,
            n_steps=ig_steps,
        )
        trunc_seq = row["sequence"][: sample_embedding.shape[1]]
        ig_df = residue_importance_dataframe(trunc_seq, residue_attrs.squeeze(0).numpy()[: len(trunc_seq)])
        st.session_state.predict_run["inspected_result"] = {
            "explain_idx": explain_idx,
            "trunc_seq": trunc_seq,
            "ig_df": ig_df,
        }

    inspected_result = st.session_state.predict_run.get("inspected_result")
    if inspected_result is None:
        st.info("Select a sequence and click Inspect sequence to run explainability.")
    else:
        explain_idx = inspected_result["explain_idx"]
        row = df_valid.iloc[explain_idx]
        trunc_seq = inspected_result["trunc_seq"]
        ig_df = inspected_result["ig_df"]

        st.markdown(
            f"**Predicted class:** {pred_table.iloc[explain_idx]['predicted_class']}  |  "
            f"**Confidence:** {confs[explain_idx]:.3f}"
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

        if bundle.uses_attention and attn is not None:
            attn_vec = attn[explain_idx].numpy()[: len(trunc_seq)]
            attn_df = attention_dataframe(trunc_seq, attn_vec)
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
            with st.spinner("Running ESMFold..."):
                pdb_path = infer_structure_with_esmfold(row["sequence"], Path(tempfile.gettempdir()) / "protein_sequence_app_v2")
            if pdb_path is None:
                st.error("ESMFold inference is unavailable in this environment. Configure dependencies and retry.")
            else:
                show_structure_viewer(pdb_path, residue_importance=ig_df, style_mode=structure_style)
                st.download_button("Download PDB", pdb_path.read_bytes(), file_name=f"{row['seq_id']}.pdb", mime="chemical/x-pdb")
