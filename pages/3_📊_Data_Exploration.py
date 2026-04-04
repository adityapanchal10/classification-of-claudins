import numpy as np
import pandas as pd
import streamlit as st
import torch

from core.io_utils import detect_input_dataframe, validate_sequences
from core.ui import global_sidebar
from core.visuals import visualize_sequence_residue_embeddings

global_sidebar()

st.title("Data Exploration")

# Refresh this page's charts when the app theme changes.
theme_type = str(getattr(getattr(st.context, "theme", None), "type", "light")).lower()
theme_state_key = "_data_exploration_active_theme"
last_theme = st.session_state.get(theme_state_key)
if last_theme is None:
    st.session_state[theme_state_key] = theme_type
elif last_theme != theme_type:
    st.session_state[theme_state_key] = theme_type
    st.rerun()

# Check for pre-stored sequences and embeddings from Predict page
pre_stored_df = st.session_state.get("input_sequences_df", None)
pre_stored_embeddings = st.session_state.get("generated_embeddings", None)

# Sequence input section
st.subheader("Sequence Input")

# Initialize variables
df = None
embeddings = None

if pre_stored_df is not None:
    st.info("📌 Using sequences and embeddings from Predict page")
    use_pre_stored = st.checkbox("Use pre-stored data", value=True)
    if use_pre_stored:
        df = pre_stored_df.copy()
        embeddings = pre_stored_embeddings  # Store reference, no need to clone
    else:
        # Allow manual input override
        uploaded = st.file_uploader("Upload FASTA for exploration", type=["fasta", "fa", "faa", "txt"])
        text_value = st.text_area("Or paste FASTA / one-sequence-per-line text", height=140)
        if uploaded is not None or text_value.strip():
            df = validate_sequences(detect_input_dataframe(text_value, uploaded))
else:
    uploaded = st.file_uploader("Upload FASTA for exploration", type=["fasta", "fa", "faa", "txt"])
    text_value = st.text_area("Or paste FASTA / one-sequence-per-line text", height=140)
    if uploaded is not None or text_value.strip():
        df = validate_sequences(detect_input_dataframe(text_value, uploaded))

if df is not None and not df.empty:
    # Embedding visualization section
    st.subheader("Embedding Visualization")
    
    # If embeddings not already loaded from Predict page, offer file upload
    if embeddings is None:
        embeddings_file = st.file_uploader(
            "Upload embeddings (NPY or PT format)",
            type=["npy", "pt", "pth"],
            key="embeddings_upload"
        )
        
        if embeddings_file is not None:
            try:
                if embeddings_file.name.endswith(".npy"):
                    embeddings = np.load(embeddings_file)
                elif embeddings_file.name.endswith((".pt", ".pth")):
                    embeddings = torch.load(embeddings_file, map_location="cpu")
                else:
                    st.error("Unsupported file format. Use .npy or .pt/.pth")
                    embeddings = None
            except Exception as e:
                st.error(f"Error loading embeddings: {str(e)}")
                embeddings = None
    
    if embeddings is not None:
        # Validate embeddings shape
        if hasattr(embeddings, "shape"):
            E_shape = embeddings.shape
        else:
            E_shape = None
        
        if E_shape is None or len(E_shape) != 3:
            st.error(f"Expected embeddings shape (num_sequences, num_residues, embedding_dim), got {E_shape}")
        else:
            N, R, D = E_shape
            st.info(f"Loaded embeddings: {N} sequences × {R} residues × {D} dimensions")

            sequence_count = min(len(df), N)
            sequence_labels = [
                f"{df.iloc[i]['seq_id']} ({df.iloc[i]['length']} aa)"
                for i in range(sequence_count)
            ]
            default_selection = sequence_labels.copy()

            # Add frosted sticky styling for visualization controls.
            st.markdown(
                """
                <style>
                [data-testid="stMainBlockContainer"] {
                    overflow-y: auto;
                }
                div:has(> [data-testid="stMultiSelect"]) {
                    position: sticky;
                    top: 0;
                    z-index: 999;
                    padding: 0.9rem 1rem;
                    margin: 0 0 0.75rem 0;
                    border-radius: 16px;
                    background:
                        linear-gradient(135deg, rgba(255, 255, 255, 0.24), rgba(255, 255, 255, 0.08)),
                        rgba(20, 24, 32, 0.10);
                    border: 1px solid rgba(255, 255, 255, 0.18);
                    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
                    backdrop-filter: blur(14px) saturate(160%);
                    -webkit-backdrop-filter: blur(14px) saturate(160%);
                }
                @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
                    div:has(> [data-testid="stMultiSelect"]) {
                        background: rgba(248, 250, 252, 0.82);
                    }
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            selected_labels = st.multiselect(
                "Sequences to visualize",
                options=sequence_labels,
                default=default_selection,
                help="Select or deselect sequences to update the embedding distributions.",
            )

            with st.columns([1, 5])[0]:
                n_pcs = st.number_input(
                    "# of PCA components",
                    min_value=1,
                    max_value=min(3, D),
                    value=min(3, D),
                    step=1,
                )
            show_pca_btn = st.button('Show pca distribution', type="primary")
            viz_mode = "pca"

            st.divider()

            if not selected_labels:
                st.info("Select at least one sequence to visualize embeddings.")
            else:
                label_to_idx = {label: idx for idx, label in enumerate(sequence_labels)}
                selected_indices = [label_to_idx[label] for label in selected_labels]
                filtered_df = df.iloc[selected_indices].reset_index(drop=True)

                if hasattr(embeddings, "detach"):
                    filtered_embeddings = embeddings[selected_indices]
                else:
                    filtered_embeddings = embeddings[selected_indices, :, :]

                residues_list = [list(seq) for seq in filtered_df["sequence"].values]
                ids_list = filtered_df["seq_id"].tolist()

                if show_pca_btn:
                    with st.spinner("Generating visualizations..."):
                        visualize_sequence_residue_embeddings(
                            ids=ids_list,
                            residues=residues_list,
                            embeddings=filtered_embeddings,
                            max_plot_sequences=len(selected_indices),
                            mode=viz_mode,
                            n_pcs=n_pcs,
                        )
