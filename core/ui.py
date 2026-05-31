import streamlit as st
import uuid
import importlib
import os

try:
    _psutil_spec = importlib.util.find_spec("psutil")
    psutil = importlib.import_module("psutil") if _psutil_spec is not None else None
except Exception:
    psutil = None

from core.config import MODEL_REGISTRY
from core.embeddings import DEFAULT_EMBEDDER_NAME, available_embedder_names, embedder_supports_msa_mode


DEFAULT_SEQ_LENGTH = 190
DEFAULT_BATCH_SIZE = 64
DEFAULT_IG_STEPS = 50
DEFAULT_ENABLE_MEMORY_LOGS = False
DEFAULT_EMBED_IN_MSA_MODE = True


def app_header():
    st.title("Claudin Classification and Explainability with ESM Embeddings")
    st.caption("Pretrained embedding -> model selection -> prediction -> explainability -> structure")


def _start_cache_trace_run():
    if "_cache_trace_session_id" not in st.session_state:
        st.session_state["_cache_trace_session_id"] = uuid.uuid4().hex[:8]
    run_idx = int(st.session_state.get("_cache_trace_run_idx", 0)) + 1
    st.session_state["_cache_trace_run_idx"] = run_idx
    st.session_state["_cache_trace_current"] = f"{st.session_state['_cache_trace_session_id']}-r{run_idx}"


def cache_log(message: str, once_key: str | None = None):
    trace = st.session_state.get("_cache_trace_current", "no-trace")
    if once_key is not None:
        seen = st.session_state.get("_cache_log_once_seen", set())
        if once_key in seen:
            return
        seen.add(once_key)
        st.session_state["_cache_log_once_seen"] = seen

    full_message = f"[CACHE][{trace}] {message}"
    if st.session_state.get("_cache_log_last") == full_message:
        return
    st.session_state["_cache_log_last"] = full_message
    print(full_message)


def memory_log(step: str):
    if not bool(st.session_state.get("global_enable_memory_logs", DEFAULT_ENABLE_MEMORY_LOGS)):
        return

    trace = st.session_state.get("_cache_trace_current", "no-trace")
    if psutil is not None:
        try:
            rss_bytes = psutil.Process(os.getpid()).memory_info().rss
            total_bytes = psutil.virtual_memory().total
            rss_mb = rss_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)

            prev_rss_mb = st.session_state.get("_mem_log_prev_rss_mb")
            delta_mb = 0.0 if prev_rss_mb is None else (rss_mb - prev_rss_mb)
            st.session_state["_mem_log_prev_rss_mb"] = rss_mb

            msg = f"[MEM][{trace}] {step} rss_mb={rss_mb:.1f}/{total_mb:.1f} delta_mb={delta_mb:+.1f}"
        except Exception:
            msg = f"[MEM][{trace}] {step} rss_mb=unavailable/unavailable delta_mb=unavailable"
    else:
        msg = f"[MEM][{trace}] {step} rss_mb=unavailable/unavailable delta_mb=unavailable"

    if st.session_state.get("_mem_log_last") == msg:
        return
    st.session_state["_mem_log_last"] = msg
    print(msg)


def initialize_session_cache_state():
    if st.session_state.get("_session_initialized", False):
        return

    # Fresh browser sessions (e.g., F5/Ctrl+R) should start from a clean cache state.
    try:
        st.cache_data.clear()
        cache_log("Cleared st.cache_data")
    except Exception:
        cache_log("Failed to clear st.cache_data", once_key="cache_data_clear_failed")
        pass
    try:
        st.cache_resource.clear()
        cache_log("Cleared st.cache_resource")
    except Exception:
        cache_log("Failed to clear st.cache_resource", once_key="cache_resource_clear_failed")
        pass

    st.session_state["_session_initialized"] = True
    cache_log("Session cache initialized", once_key="session_cache_initialized")


def global_sidebar():
    _start_cache_trace_run()
    initialize_session_cache_state()
    all_models = list(MODEL_REGISTRY.keys())
    default_model = all_models[0]
    if st.session_state.get("global_model_name") not in all_models:
        st.session_state["global_model_name"] = default_model
    if "global_ig_steps" not in st.session_state:
        st.session_state["global_ig_steps"] = DEFAULT_IG_STEPS
    if "global_enable_memory_logs" not in st.session_state:
        st.session_state["global_enable_memory_logs"] = DEFAULT_ENABLE_MEMORY_LOGS
    if "global_embedder_name" not in st.session_state:
        st.session_state["global_embedder_name"] = DEFAULT_EMBEDDER_NAME
    if "global_embed_in_msa_mode" not in st.session_state:
        st.session_state["global_embed_in_msa_mode"] = DEFAULT_EMBED_IN_MSA_MODE

    previous_model = st.session_state.get("_prev_model_name")
    previous_embedder_name = st.session_state.get("_prev_embedder_name")
    previous_msa_only = st.session_state.get("_prev_embed_in_msa_mode")

    st.sidebar.header("Global settings")
    embedder_options = available_embedder_names()
    
    # Read current embedder preference from session state
    current_embedder = st.session_state.get("global_embedder_name", DEFAULT_EMBEDDER_NAME)
    if current_embedder not in embedder_options:
        current_embedder = DEFAULT_EMBEDDER_NAME
    
    # Find the index to display
    try:
        embedder_index = embedder_options.index(current_embedder)
    except ValueError:
        embedder_index = 0
    
    # Create selectbox without key, manually handling state
    emb = st.sidebar.selectbox(
        "Embedder",
        embedder_options,
        index=embedder_index,
    )
    
    # Update session state if embedder changed
    if emb != current_embedder:
        st.session_state["global_embedder_name"] = emb
    
    msa_supported = embedder_supports_msa_mode(emb)
    
    # Read the current preference from session state
    current_msa_preference = st.session_state.get("global_embed_in_msa_mode", DEFAULT_EMBED_IN_MSA_MODE)
    
    # Create the toggle without a key, manually handling the state
    msa_toggle_value = st.sidebar.toggle(
        "Embed in MSA mode",
        value=current_msa_preference,
        disabled=not msa_supported,
    )
    
    # Determine the effective MSA mode:
    # If the embedder doesn't support MSA, force it to False regardless of toggle
    if not msa_supported:
        msa_only = False
    else:
        msa_only = msa_toggle_value
    
    # Update session state only if the effective value changed
    if msa_only != current_msa_preference:
        st.session_state["global_embed_in_msa_mode"] = msa_only
    # Filter available models to those compatible with the selected embedder.
    model_options = []
    for mn, meta in MODEL_REGISTRY.items():
        compat = meta.get("compatible_embedder")
        if isinstance(compat, (list, tuple)):
            if emb in compat:
                model_options.append(mn)
        else:
            if compat == emb:
                model_options.append(mn)

    if not model_options:
        # Fallback: if no explicit compatible model is found, show all models.
        model_options = all_models
        st.sidebar.info(f"No models found compatible with {emb}; showing all models.")

    # Read current model preference from session state
    current_model = st.session_state.get("global_model_name")
    if current_model not in model_options:
        current_model = model_options[0]
    
    # Find the index to display
    try:
        model_index = model_options.index(current_model)
    except ValueError:
        model_index = 0
    
    # Create selectbox without key, manually handling state
    model_name = st.sidebar.selectbox(
        "Model",
        model_options,
        index=model_index,
    )
    
    # Update session state if model changed
    if model_name != current_model:
        st.session_state["global_model_name"] = model_name
    
    ig_steps = st.sidebar.slider("Integrated Gradients steps", min_value=50, max_value=200, step=10, key="global_ig_steps")

    # When the classifier changes, discard stale model-specific results.
    # Embeddings and input data are model-independent (from ESM) and kept.
    if previous_model is not None and model_name != previous_model:
        for key in ("predict_run",):
            st.session_state.pop(key, None)
        cache_log(f"Model changed {previous_model} -> {model_name}; cleared prediction state")
    st.session_state["_prev_model_name"] = model_name

    if previous_embedder_name is not None and emb != previous_embedder_name:
        for key in (
            "generated_embeddings",
            "generated_embeddings_embedder",
            "generated_embeddings_msa_only",
            "predict_run",
            "compare_embeddings_msa_only",
        ):
            st.session_state.pop(key, None)
        cache_log(f"Embedder changed {previous_embedder_name} -> {emb}; cleared embeddings and prediction state")
    st.session_state["_prev_embedder_name"] = emb

    if previous_msa_only is not None and msa_only != previous_msa_only:
        for key in (
            "generated_embeddings",
            "generated_embeddings_embedder",
            "generated_embeddings_msa_only",
            "predict_run",
            "compare_embeddings_msa_only",
        ):
            st.session_state.pop(key, None)
        cache_log(f"Embed in MSA mode changed {previous_msa_only} -> {msa_only}; cleared embeddings and prediction state")
    st.session_state["_prev_embed_in_msa_mode"] = msa_only

    st.sidebar.markdown(
        "<hr style='margin:0.35rem 0 0 0; border:0; border-top:1px solid rgba(156,163,175,0.35);' />",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<p style='margin:0.6rem 0 0.1rem 0; font-size:0.68rem; color:#9CA3AF;'>Diagnostics</p>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebarUserContent"] .st-key-global_enable_memory_logs {
            margin-top: -0.6rem;
            margin-bottom: 0;
            padding-top: 0;
            padding-bottom: 0;
        }
        [data-testid="stSidebarUserContent"] .st-key-global_enable_memory_logs [data-baseweb="checkbox"] > div {
            transform: scale(0.82);
            transform-origin: left center;
        }
        [data-testid="stSidebarUserContent"] .st-key-global_enable_memory_logs [data-testid="stCheckbox"] p {
            font-size: 0.82rem;
            color: #9CA3AF;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.checkbox(
        "Enable memory logs",
        key="global_enable_memory_logs",
        help="Print minimal memory snapshots at major action completion points.",
    )
    return model_name, DEFAULT_SEQ_LENGTH, DEFAULT_BATCH_SIZE, ig_steps


def toast_once(session_key, item_key, message):
    toast_state = st.session_state.get(session_key, {})
    if toast_state.get(item_key, False):
        return False

    st.toast(message)
    toast_state[item_key] = True
    st.session_state[session_key] = toast_state
    return True
