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
    # "Transformer + MLP": "",
    # "Transformer + MLP (ECS only)": "",
    "Transformer + MLP 2": "https://drive.google.com/file/d/15UgFUV9prPKHVo2LbJpGlErlKQPagNfs/view?usp=drive_link",
    "Transformer + MLP 2 (ECS only)": "https://drive.google.com/file/d/1hOzKZiQMtQwoIgy-iZiu7sQ9xdmin8it/view?usp=drive_link",
    "Transformer + MLP 2 Non-MSA": "https://drive.google.com/file/d/1DapCccRHCPQ1GMOQnB9e0dY75Mf2gIZu/view?usp=drive_link",
    "Transformer + MLP 2 Non-MSA (ECS only)": "https://drive.google.com/file/d/1bAP-orO6DyZegiZT5j_lWKRtYnqQ3NrT/view?usp=drive_link",
    "Transformer + MLP 2 ESM2": "https://drive.google.com/file/d/1_HGkFovrdWqlNGus_gyjIrjKPThtW-W1/view?usp=drive_link",
    "Transformer + MLP 2 ESM2 (ECS only)": "https://drive.google.com/file/d/1qsm52YGMTqQNqMoUVBA4kI72EUzU41WP/view?usp=drive_link",
    # "Simple Linear": "",
    "Simple Linear 2": "https://drive.google.com/file/d/1aOs2OudahQK2RgC0qja4pKMw7KtIGOwm/view?usp=drive_link",
    "Simple Linear 2 (ECS only)": "https://drive.google.com/file/d/1oYsjei345Ynmy_Pv57oqLRUfhEp4SlKH/view?usp=drive_link",
    "Simple Linear 2 Non-MSA": "https://drive.google.com/file/d/1BB8b8A4a4JIFbMrVHlPwPWNtHgkkguhs/view?usp=drive_link",
    "Simple Linear 2 Non-MSA (ECS only)": "https://drive.google.com/file/d/1MIvQTWUG0Upyb0a_-KG8SEPP-mYTPGqt/view?usp=drive_link",
    "Simple Linear 2 ESM2": "https://drive.google.com/file/d/1fbh1VX2v18gH1cy4K5r4Tes678sV395y/view?usp=drive_link",
    "Simple Linear 2 Diverse": "https://drive.google.com/file/d/1iaLFkcQFKMH1dBWjYKyBMzDIN1BguM5n/view?usp=sharing",
    "Simple Linear 2 Diverse (ECS only)": "https://drive.google.com/file/d/1BBzItiIZK-GJ_sz5kde1Rf3D3isHr7f0/view?usp=sharing",
    "Simple Linear 2 Balanced": "https://drive.google.com/file/d/1x-UFJ1j0_Hb9Mi_gynmCsRg3esaITT2i/view?usp=drive_link",
    "Simple Linear 2 Balanced (ECS only)": "https://drive.google.com/file/d/1mQ_xk9s4NtP8lVeY3pixhRs1Cl8uPun0/view?usp=drive_link",
    # "Simple CNN": "",
    # "Transformer (simple)": "",
    # "Transformer (complex)": "",
}

# Optional override via environment variable.
# Example:
# {"Transformer + MLP": "https://drive.google.com/file/d/<id>/view", ...}
CHECKPOINT_GDRIVE_URLS.update(_load_json_mapping("CHECKPOINT_GDRIVE_URLS_JSON"))


def resolve_checkpoint_url(model_name: str, checkpoint_file: Optional[str] = None) -> Optional[str]:
    mapped_url = CHECKPOINT_GDRIVE_URLS.get(model_name, "").strip()
    if mapped_url:
        return mapped_url

    # Backward compatibility: also support checkpoint filename keys in custom env mappings.
    if checkpoint_file:
        legacy_url = CHECKPOINT_GDRIVE_URLS.get(checkpoint_file, "").strip()
        if legacy_url:
            return legacy_url
    return None

CLASS_MAP = {0: "Barrier forming", 1: "Cation-channel forming", 2: "Anion-channel forming"}
DEFAULT_CLASSES = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]

MODEL_REGISTRY = {
    # "Transformer + MLP": {
    #     "class_name": "Transformer + MLP Classifier",
    #     "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on complete data (individual claudin FASTAs splitted from big FASTA) and validated via LOFO CV and Grouped Holdout",
    #     "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
    #     "uses_attention": True,
    #     "checkpoint_file": "transformer_mlp_classifier.pt",
    #     "compatible_embedder": "MSA Transformer",
    #     "kwargs": {
    #         "embedding_dim": 768,
    #         "proj_dim": 128,
    #         "num_classes": 3,
    #         "num_heads": 4,
    #         "num_attention_blocks": 1,
    #         "dropout": 0.4,
    #         "seq_len": 190,
    #     },
    # },
    # "Transformer + MLP (ECS only)": {
    #     "class_name": "Transformer + MLP Classifier",
    #     "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on complete data (individual claudin FASTAs splitted from big FASTA and modified to contain only the ECS1/2 segments) and validated via LOFO CV and Grouped Holdout",
    #     "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
    #     "uses_attention": True,
    #     "checkpoint_file": "transformer_mlp_classifier_ecs_only.pt",
    #     "residue_slice": [(27, 81), (138, 164)],
    #     "compatible_embedder": "MSA Transformer",
    #     "kwargs": {
    #         "embedding_dim": 768,
    #         "proj_dim": 128,
    #         "num_classes": 3,
    #         "num_heads": 4,
    #         "num_attention_blocks": 1,
    #         "dropout": 0.4,
    #         "seq_len": 80,
    #     },
    # },
    "Transformer + MLP 2": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on train/val split from the single FASTA file",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_single_fasta.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": {
            "embedding_dim": 768,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 220,
        },
    },
    "Transformer + MLP 2 (ECS only)": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on train/val split from the single FASTA file modified to contain only the ECS1/2 segments",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_single_fasta_ecs_only.pt",
        "compatible_embedder": "MSA Transformer",
        "residue_slice": [(27, 81), (138, 164)],
        "kwargs": {
            "embedding_dim": 768,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 80,
        },
    },
    "Transformer + MLP 2 Non-MSA": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on train/val split from the single FASTA file, embeddings generated without MSA context.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_non_msa_single_fasta.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": {
            "embedding_dim": 768,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 220,
        },
    },
    "Transformer + MLP 2 Non-MSA (ECS only)": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on train/val split from the single FASTA file modified to contain only the ECS1/2 segments.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_non_msa_single_fasta_ecs_only.pt",
        "compatible_embedder": "MSA Transformer",
        "residue_slice": [(27, 81), (138, 164)],
        "kwargs": {
            "embedding_dim": 768,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 80,
        },
    },
    "Transformer + MLP 2 ESM2": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with ESM2 positional embeddings and fused pooled sequence features. Trained on train/val split from the single FASTA file.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_esm2_single_fasta.pt",
        "compatible_embedder": "ESM2",
        "kwargs": {
            "embedding_dim": 640,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 220,
        },
    },
    "Transformer + MLP 2 ESM2 (ECS only)": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with ESM2 positional embeddings and fused pooled sequence features. Trained on train/val split from the single FASTA file modified to contain only the ECS1/2 segments.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_esm2_single_fasta_ecs_only.pt",
        "residue_slice": [(27, 81), (138, 164)],
        "compatible_embedder": "ESM2",
        "kwargs": {
            "embedding_dim": 640,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 80,
        },
    },
    # "Simple Linear": {
    #     "class_name": "Simple Linear Classifier",
    #     "description": "LayerNorm baseline using learned residue attention and a single linear head.",
    #     "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
    #     "uses_attention": False,
    #     "checkpoint_file": "simple_linear_classifier.pt",
    #     "compatible_embedder": "MSA Transformer",
    #     "kwargs": { 
    #         "n_classes": 3, 
    #         "dropout": 0.2,
    #         "embedding_dim": 768
    #     },
    # },
    "Simple Linear 2": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_single_fasta.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    "Simple Linear 2 (ECS only)": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file, containing only the ECS1/2 segments.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_single_fasta_ecs_only.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    "Simple Linear 2 Non-MSA": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file, embeddings generated without MSA context.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_non_msa_single_fasta.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    "Simple Linear 2 Non-MSA (ECS only)": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file modified to contain only the ECS1/2 segments, embeddings generated without MSA context.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_non_msa_single_fasta_ecs_only.pt",
        "compatible_embedder": "MSA Transformer",
        "residue_slice": [(27, 81), (138, 164)],
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    "Simple Linear 2 ESM2": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file using ESM2 embeddings.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_esm2.pt",
        "compatible_embedder": "ESM2",
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 640
        },
    },
    "Simple Linear 2 Diverse": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_single_fasta_diverse.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    "Simple Linear 2 Diverse (ECS only)": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file modified to contain only the ECS1/2 segments, embeddings generated without MSA context.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_single_fasta_diverse_ecs_only.pt",
        "compatible_embedder": "MSA Transformer",
        "residue_slice": [(27, 81), (138, 164)],
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    "Simple Linear 2 Balanced": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_single_fasta_balanced.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    "Simple Linear 2 Balanced (ECS only)": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file modified to contain only the ECS1/2 segments, embeddings generated without MSA context.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_single_fasta_balanced_ecs_only.pt",
        "compatible_embedder": "MSA Transformer",
        "residue_slice": [(27, 81), (138, 164)],
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    # "Simple CNN": {
    #     "class_name": "Simple CNN Classifier",
    #     "description": "Parallel CNN model that captures local motifs with multiple kernel sizes.",
    #     "architecture": "LayerNorm -> parallel Conv2d kernels -> ReLU -> global max pooling -> concatenate -> dropout -> linear classifier",
    #     "uses_attention": False,
    #     "checkpoint_file": "simple_cnn_classifier.pt",
    #     "compatible_embedder": "MSA Transformer",
    #     "kwargs": {
    #         "n_classes": 3,
    #         "embedding_dim": 768,
    #         "n_filters": 100,
    #         "filter_sizes": [3, 4, 5],
    #         "dropout": 0.1,
    #     },
    # },
    # "Transformer (simple)": {
    #     "class_name": "Transformer Classifier (simple)",
    #     "description": "Transformer encoder with learned positional embeddings and mean pooling.",
    #     "architecture": "Positional embedding add -> TransformerEncoder layers -> mean pooling -> 2-layer MLP classifier",
    #     "uses_attention": True,
    #     "checkpoint_file": "transformer_classifier_simple.pt",
    #     "compatible_embedder": "MSA Transformer",
    #     "kwargs": {
    #         "embedding_dim": 768,
    #         "num_classes": 3,
    #         "num_heads": 8,
    #         "num_layers": 2,
    #         "dropout": 0.1,
    #         "max_seq_len": 512,
    #     },
    # },
    # "Transformer (complex)": {
    #     "class_name": "Transformer Classifier (complex)",
    #     "description": "Residual 1D-convolution model with attention pooling.",
    #     "architecture": "Input projection -> positional embedding projection -> residual Conv1d blocks -> attention pooling -> MLP classifier",
    #     "uses_attention": True,
    #     "checkpoint_file": "transformer_classifier_complex.pt",
    #     "compatible_embedder": "MSA Transformer",
    #     "kwargs": {
    #         "embedding_dim": 768,
    #         "num_classes": 3,
    #         "max_len": 512,
    #         "hidden_dim": 128,
    #         "pos_emb_dim": 128,
    #         "num_conv_blocks": 3,
    #         "kernel_size": 3,
    #         "dropout": 0.2,
    #         "use_pc_features": False,
    #         "pc_dim": 3,
    #         "pc_mlp_dim": 32,
    #     },
    # },
}