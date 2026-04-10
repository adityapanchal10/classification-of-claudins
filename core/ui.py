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


DEFAULT_SEQ_LENGTH = 190
DEFAULT_BATCH_SIZE = 64
DEFAULT_IG_STEPS = 50
DEFAULT_ENABLE_MEMORY_LOGS = False


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
    model_options = list(MODEL_REGISTRY.keys())
    default_model = model_options[0]
    if st.session_state.get("global_model_name") not in model_options:
        st.session_state["global_model_name"] = default_model
    if "global_ig_steps" not in st.session_state:
        st.session_state["global_ig_steps"] = DEFAULT_IG_STEPS
    if "global_enable_memory_logs" not in st.session_state:
        st.session_state["global_enable_memory_logs"] = DEFAULT_ENABLE_MEMORY_LOGS

    st.sidebar.header("Global settings")
    model_name = st.sidebar.selectbox("Model", model_options, key="global_model_name")
    ig_steps = st.sidebar.slider("Integrated Gradients steps", min_value=50, max_value=200, step=10, key="global_ig_steps")

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
