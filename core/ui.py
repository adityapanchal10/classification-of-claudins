import streamlit as st
import uuid

from core.config import MODEL_REGISTRY


DEFAULT_SEQ_LENGTH = 190
DEFAULT_BATCH_SIZE = 64
DEFAULT_IG_STEPS = 50


def app_header():
    st.title("Claudin Classification and Explainability with MSA Transformer Embeddings")
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

    for key in (
        "_predict_embedding_cache_key",
        "_predict_embedding_cache_value",
        "_predict_baseline_cache_key",
        "_predict_baseline_cache_value",
        "_compare_embedding_cache_key",
        "_compare_embedding_cache_value",
        "run_data_exploration",
    ):
        removed = st.session_state.pop(key, None)
        if removed is not None:
            cache_log(f"Cleared session key={key}")

    st.session_state["_session_initialized"] = True
    cache_log("Session cache initialized", once_key="session_cache_initialized")


def global_sidebar():
    _start_cache_trace_run()
    initialize_session_cache_state()
    model_options = list(MODEL_REGISTRY.keys())
    default_model = model_options[0]
    if st.session_state.get("global_model_name") not in model_options:
        st.session_state["global_model_name"] = default_model
    if "global_ig_steps" not in st.session_state:
        st.session_state["global_ig_steps"] = DEFAULT_IG_STEPS

    st.sidebar.header("Global settings")
    model_name = st.sidebar.selectbox("Model", model_options, key="global_model_name")
    ig_steps = st.sidebar.slider("Integrated Gradients steps", min_value=50, max_value=200, step=10, key="global_ig_steps")
    return model_name, DEFAULT_SEQ_LENGTH, DEFAULT_BATCH_SIZE, ig_steps


def toast_once(session_key, item_key, message):
    toast_state = st.session_state.get(session_key, {})
    if toast_state.get(item_key, False):
        return False

    st.toast(message)
    toast_state[item_key] = True
    st.session_state[session_key] = toast_state
    return True
