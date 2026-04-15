import streamlit as st

from core.embeddings import get_embedder
from core.io_utils import detect_input_dataframe, validate_sequences
from core.ui import cache_log, global_sidebar, memory_log, toast_once
from core.visuals import visualize_sequence_residue_embeddings

st.set_page_config(page_title="Data Exploration", layout="wide", page_icon="🧬")
st.logo("🧬")


def _infer_embedding_params(df_valid):
    seq_length = int(df_valid["length"].max())
    batch_size = min(len(df_valid), 32)

    return seq_length, batch_size

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
using_pre_stored_data = False

if pre_stored_df is not None:
    st.info("📌 Using sequences and embeddings from Predict page")
    use_pre_stored = st.checkbox("Use pre-stored data", value=True)
    if use_pre_stored:
        df = pre_stored_df.copy()
        embeddings = pre_stored_embeddings
        using_pre_stored_data = True
    else:
        # Allow manual input override
        uploaded = st.file_uploader("Upload FASTA for exploration", type=["fasta", "fa", "faa", "txt"])
        text_value = st.text_area("**OR** paste FASTA / one-sequence-per-line text", height=140, placeholder=">seq1\nMKT...\n>seq2\nVVV...")
        if uploaded is not None or text_value.strip():
            df = validate_sequences(detect_input_dataframe(text_value, uploaded))
else:
    uploaded = st.file_uploader("Upload FASTA for exploration", type=["fasta", "fa", "faa", "txt"])
    text_value = st.text_area("**OR** paste FASTA / one-sequence-per-line text", height=140, placeholder=">seq1\nMKT...\n>seq2\nVVV...")
    if uploaded is not None or text_value.strip():
        df = validate_sequences(detect_input_dataframe(text_value, uploaded))

run_exploration = using_pre_stored_data or st.button("Run exploration", type="primary")
if run_exploration and not using_pre_stored_data:
    print("[PAGE Explore] Run exploration")

if run_exploration:
    if df is None or df.empty:
        st.warning("Provide sequence input via textbox or file upload.")
        st.stop()

    df_valid = df[df["is_valid"]].copy() if "is_valid" in df.columns else df.copy()
    if df_valid.empty:
        st.error("No valid amino acid sequences were found.")
        st.stop()

    # Embedding visualization section
    st.subheader("Embedding Visualization")

    # If embeddings are not already available, generate them from current valid sequences.
    if embeddings is None:
        try:
            with st.spinner("Generating embeddings for input sequences..."):
                embedder = get_embedder()
                embedder_name = getattr(embedder, "model_name", "esm_msa1b_t12_100M_UR50S")
                toast_once("_embedder_ready_toast_shown", embedder_name, f"⚗️ Embedder ready: {embedder_name}")
                seq_length, batch_size = _infer_embedding_params(df_valid)
                embeddings = embedder.embed_sequences_per_residue(
                    df_valid["sequence"].tolist(),
                    seq_length=seq_length,
                    batch_size=batch_size,
                )
                cache_log("explore cache miss for embeddings; generated fresh embeddings")
                st.caption(f"Embedding params: seq_length={seq_length}, batch_size={batch_size}")
                print(f"[PAGE Explore] Embeddings ready n_seq={len(df_valid)}")
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
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

            sequence_count = min(len(df_valid), N)
            sequence_labels = [
                f"{df_valid.iloc[i]['seq_id']} ({df_valid.iloc[i]['length']} aa)"
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

            st.markdown(
                "<p style='margin-top:-0.25rem; margin-bottom:0.7rem; color:#94a3b8; font-size:0.85rem;'>"
                "The PCA plots are built only from the sequences selected above. If the selection is changed, the plots can "
                "look different. For a decent comparison, keep the same selected list and use Plotly toggles to focus on "
                "sequences of interest. Use a smaller subset when you want to closely compare a few similar sequences, so "
                "the patterns are easier to see."
                "</p>",
                unsafe_allow_html=True,
            )

            with st.columns([1, 5])[0]:
                n_pcs = st.number_input(
                    "# of PCA components",
                    min_value=1,
                    max_value=min(3, D),
                    value=min(3, D),
                    step=1,
                )
            show_pca_btn = st.button('Show pca distribution')
            viz_mode = "pca"

            st.divider()

            if not selected_labels:
                st.info("Select at least one sequence to visualize embeddings.")
            else:
                label_to_idx = {label: idx for idx, label in enumerate(sequence_labels)}
                selected_indices = [label_to_idx[label] for label in selected_labels]
                filtered_df = df_valid.iloc[selected_indices].reset_index(drop=True)

                if hasattr(embeddings, "detach"):
                    filtered_embeddings = embeddings[selected_indices]
                else:
                    filtered_embeddings = embeddings[selected_indices, :, :]

                residues_list = [list(seq) for seq in filtered_df["sequence"].values]
                ids_list = filtered_df["seq_id"].tolist()

                if show_pca_btn:
                    print(f"[PAGE Explore] Show PCA n_seq={len(selected_indices)} pcs={n_pcs}")
                    with st.spinner("Generating visualizations..."):
                        visualize_sequence_residue_embeddings(
                            ids=ids_list,
                            residues=residues_list,
                            embeddings=filtered_embeddings,
                            max_plot_sequences=len(selected_indices),
                            mode=viz_mode,
                            n_pcs=n_pcs,
                        )
                    memory_log("explore.show_pca.done")
