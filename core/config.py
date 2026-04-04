from pathlib import Path
import json
import os
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
IMAGES_DIR = BASE_DIR / "images"


def _load_json_mapping(env_var_name: str) -> dict[str, str]:
    raw = os.getenv(env_var_name, "").strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    mapping: dict[str, str] = {}
    for key, value in loaded.items():
        key_str = str(key).strip()
        value_str = str(value).strip()
        if key_str and value_str:
            mapping[key_str] = value_str
    return mapping


CHECKPOINT_GDRIVE_URLS = {
    "transformer_mlp_classifier.pt": "https://drive.google.com/file/d/1Ah_lZXwuqNoY9ke_Zh2iQ9FD4h5XRRpk/view?usp=drive_link",
    "simple_linear_classifier.pt": "https://drive.google.com/file/d/1yAw3_8LBGx7wkwY_GRGPPrmMWqkRrs-k/view?usp=drive_link",
    "simple_cnn_classifier.pt": "https://drive.google.com/file/d/1lALilrt0OBFKXvzRTNcwOm_rgplkNTFF/view?usp=drive_link",
    "transformer_classifier_simple.pt": "https://drive.google.com/file/d/1Zsrj8FiF1yk_W-MCPeSuyNAycJZU6zas/view?usp=drive_link",
    "transformer_classifier_complex.pt": "https://drive.google.com/file/d/1O-pv5B_H9KPOxN67YlbMzjSTnQxQxyni/view?usp=drive_link",
}

# Optional override via environment variable.
# Example:
# {"transformer_mlp_classifier.pt": "https://drive.google.com/uc?id=<id>", ...}
CHECKPOINT_GDRIVE_URLS.update(_load_json_mapping("CHECKPOINT_GDRIVE_URLS_JSON"))

# Optional fallback: map filename -> Google Drive file id.
# Example:
# {"transformer_mlp_classifier.pt": "<drive_file_id>", ...}
CHECKPOINT_GDRIVE_FILE_IDS = _load_json_mapping("CHECKPOINT_GDRIVE_FILE_IDS_JSON")


def _build_gdrive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


def resolve_checkpoint_url(checkpoint_file: str) -> Optional[str]:
    mapped_url = CHECKPOINT_GDRIVE_URLS.get(checkpoint_file, "").strip()
    if mapped_url:
        return mapped_url

    file_id = CHECKPOINT_GDRIVE_FILE_IDS.get(checkpoint_file, "").strip()
    if file_id:
        return _build_gdrive_download_url(file_id)
    return None

CLASS_MAP = {0: "Barrier forming", 1: "Cation-channel forming", 2: "Anion-channel forming"}
DEFAULT_CLASSES = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]

MODEL_REGISTRY = {
    "Transformer + MLP Classifier": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier.pt",
        "kwargs": {
            "input_dim": 768,
            "num_classes": 3,
            "num_heads": 8,
            "num_attention_blocks": 2,
            "dropout": 0.1,
            "max_residues": 1024,
        },
    },
    "Simple Linear Classifier": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier.pt",
        "kwargs": {"n_classes": 3, "dropout": 0.2},
    },
    "Simple CNN Classifier": {
        "class_name": "Simple CNN Classifier",
        "description": "Parallel CNN model that captures local motifs with multiple kernel sizes.",
        "architecture": "LayerNorm -> parallel Conv2d kernels -> ReLU -> global max pooling -> concatenate -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_cnn_classifier.pt",
        "kwargs": {
            "n_classes": 3,
            "embedding_dim": 768,
            "n_filters": 100,
            "filter_sizes": [3, 4, 5],
            "dropout": 0.1,
        },
    },
    "Transformer Classifier (simple)": {
        "class_name": "Transformer Classifier (simple)",
        "description": "Transformer encoder with learned positional embeddings and mean pooling.",
        "architecture": "Positional embedding add -> TransformerEncoder layers -> mean pooling -> 2-layer MLP classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_classifier_simple.pt",
        "kwargs": {
            "embedding_dim": 768,
            "num_classes": 3,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "max_seq_len": 512,
        },
    },
    "Transformer Classifier (complex)": {
        "class_name": "Transformer Classifier (complex)",
        "description": "Residual 1D-convolution model with attention pooling.",
        "architecture": "Input projection -> positional embedding projection -> residual Conv1d blocks -> attention pooling -> MLP classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_classifier_complex.pt",
        "kwargs": {
            "embedding_dim": 768,
            "num_classes": 3,
            "max_len": 512,
            "hidden_dim": 128,
            "pos_emb_dim": 128,
            "num_conv_blocks": 3,
            "kernel_size": 3,
            "dropout": 0.2,
            "use_pc_features": False,
            "pc_dim": 3,
            "pc_mlp_dim": 32,
        },
    },
}