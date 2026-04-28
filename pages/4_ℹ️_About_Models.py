import pandas as pd
import streamlit as st
import torch
from torchinfo import summary

from core.config import MODEL_REGISTRY, CHECKPOINTS_DIR, IMAGES_DIR
from core.models import load_classifier_bundle
from core.ui import global_sidebar

st.set_page_config(page_title="About Models", layout="wide", page_icon="🧬")
st.logo("🧬")


global_sidebar()

st.title("About Models")
rows = []
for name, cfg in MODEL_REGISTRY.items():
    rows.append(
        {
            "Model": name,
            "Description": cfg["description"],
            "Architecture": cfg["architecture"],
            "Attention": cfg["uses_attention"],
        }
    )
st.dataframe(pd.DataFrame(rows), width="stretch")

st.subheader("Model Summary and Training Stats")
model_options = list(MODEL_REGISTRY.keys())
default_model = st.session_state.get("global_model_name", model_options[0])
default_index = model_options.index(default_model) if default_model in model_options else 0
selected_model = st.selectbox(
    "Select model", 
    model_options, 
    index=default_index,
    key="about_models_selectbox",
    on_change=lambda: st.session_state.update({"global_model_name": st.session_state["about_models_selectbox"]})
)

try:
    bundle = load_classifier_bundle(selected_model)
    model_stats = summary(
        bundle.classifier,
        input_size=(1, 128, 768),
        depth=3,
        verbose=0,
        col_names=("input_size", "output_size", "num_params", "trainable"),
    )
    st.code(str(model_stats), language="text")
except Exception as exc:
    st.error(f"Could not generate model summary for '{selected_model}': {exc}")

st.subheader("Training Checkpoint Metrics")
cfg = MODEL_REGISTRY[selected_model]
checkpoint_path = CHECKPOINTS_DIR / cfg["checkpoint_file"]

if checkpoint_path.exists():
    # Only extract the lightweight scalar metrics — avoid keeping the full
    # checkpoint (which includes optimizer state and can be 100+ MB) in memory.
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    except Exception as exc:
        st.error(f"Could not load checkpoint '{checkpoint_path.name}': {exc}")
        st.stop()
    metrics = {
        "epoch": checkpoint.get("epoch"),
        "val_auc": checkpoint.get("val_auc"),
        "val_loss": checkpoint.get("val_loss"),
        "val_acc": checkpoint.get("val_acc"),
    }
    del checkpoint
    import gc; gc.collect()

    col1, col2, col3, col4 = st.columns(4)

    raw_epoch = metrics["epoch"]
    col1.metric("Saved Epoch", (raw_epoch + 1) if raw_epoch is not None else "N/A")
    
    val_auc = metrics["val_auc"]
    col2.metric("Validation AUC", f"{val_auc:.3f}" if val_auc is not None else "N/A")
    
    val_loss = metrics["val_loss"]
    col3.metric("Validation Loss", f"{val_loss:.3f}" if val_loss is not None else "N/A")
    
    val_acc = metrics["val_acc"]
    if val_acc is not None:
        col4.metric("Validation Accuracy", f"{val_acc:.2f}%")
    else:
        col4.metric("Validation Accuracy", "N/A")

    # st.markdown("**Training Plots (Saved Images)**")
    checkpoint_stem = checkpoint_path.stem
    # st.caption(
    #     "Expected files: "
    #     f"{checkpoint_stem}_history.png, "
    #     f"{checkpoint_stem}_val_roc.png, "
    #     f"{checkpoint_stem}_class_errors.png"
    # )

    image_specs = [
        ("Training History", IMAGES_DIR / f"{checkpoint_stem}_history.png"),
        ("Validation ROC", IMAGES_DIR / f"{checkpoint_stem}_val_roc.png"),
        ("Class-wise Error Trends", IMAGES_DIR / f"{checkpoint_stem}_class_errors.png"),
    ]

    for title, image_path in image_specs:
        st.markdown(f"**{title}**")
        if image_path.exists():
            st.image(str(image_path), width="stretch")
        else:
            st.info(f"Image not found: {image_path.name}")
else:
    st.info(f"Checkpoint not found for selected model: {checkpoint_path.name}")
