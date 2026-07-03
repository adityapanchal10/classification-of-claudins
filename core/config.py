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
    "Transformer + MLP": "https://drive.google.com/file/d/1QGVmqZ76da6VTE8isXDZPg1wX_VGL2gA/view?usp=drive_link",
    "Transformer + MLP (ECS only)": "https://drive.google.com/file/d/1hooadDGGtuu6BcvRY34tlqURTUmqtXMH/view?usp=drive_link",
    "Transformer + MLP Non-MSA": "https://drive.google.com/file/d/1IFdeoc72dpTe4D_gTxZI6aSbDGT14EGi/view?usp=drive_link",
    "Transformer + MLP Non-MSA (ECS only)": "https://drive.google.com/file/d/1V7UGUG5JF4sBG0SfIZgMs1cz3-DXakk4/view?usp=drive_link",
    "Transformer + MLP ESM2": "https://drive.google.com/file/d/1LG1AYVl7Uw55L2NLZe_rD41lhdeOc9Wk/view?usp=drive_link",
    "Transformer + MLP ESM2 (ECS only)": "https://drive.google.com/file/d/1nHMNzLOhbxUab9jxDVPcVSLYdDXla9f8/view?usp=drive_link",
    "Transformer + MLP Diverse": "https://drive.google.com/file/d/18wVpX7hm3686sphnGSYHQOnrmgwIC2bi/view?usp=drive_link",
    "Transformer + MLP Diverse (ECS only)": "https://drive.google.com/file/d/1uM9Zcr-Wi1eoR272eKXQrUj4NkJblGLf/view?usp=drive_link",
    "Transformer + MLP Balanced": "https://drive.google.com/file/d/1nhvxqPYbGYLEnKD_ijK6uOeBs5QwdS3X/view?usp=drive_link",
    "Transformer + MLP Balanced (ECS only)": "https://drive.google.com/file/d/1nhvxqPYbGYLEnKD_ijK6uOeBs5QwdS3X/view?usp=drive_link",
    "Transformer + MLP Family": "https://drive.google.com/file/d/1kQldx07ac80MwtRQJrtOJZ6nQVSR_2iM/view?usp=drive_link",
    "Transformer + MLP Family (ECS only)": "https://drive.google.com/file/d/1kQldx07ac80MwtRQJrtOJZ6nQVSR_2iM/view?usp=drive_link",
    # "Simple Linear": "",
    "Simple Linear": "https://drive.google.com/file/d/1SjO58YugElFbhskNqs2L_V-VYYl8Nv-k/view?usp=drive_link",
    "Simple Linear (ECS only)": "https://drive.google.com/file/d/1GK8T3x_KiaMWVRJfjfNeEUVsi0byPQ4D/view?usp=drive_link",
    "Simple Linear Non-MSA": "https://drive.google.com/file/d/1DFUr9Mrrfq92u64buc_ZGE0BKOhEueaa/view?usp=drive_link",
    "Simple Linear Non-MSA (ECS only)": "https://drive.google.com/file/d/1U352kMRjMHMUZ1lW7SyD_pHftt-fGBv8/view?usp=drive_link",
    "Simple Linear ESM2": "https://drive.google.com/file/d/1lYnWKunw8U12O2wfThSivyhlo3RRMHtb/view?usp=drive_link",
    "Simple Linear ESM2 (ECS only)": "https://drive.google.com/file/d/1951CTCooH5ZGgQ3_zVSqWTXE-J9AK_-s/view?usp=drive_link",
    "Simple Linear Diverse": "https://drive.google.com/file/d/1iaLFkcQFKMH1dBWjYKyBMzDIN1BguM5n/view?usp=sharing",
    "Simple Linear Diverse (ECS only)": "https://drive.google.com/file/d/1BBzItiIZK-GJ_sz5kde1Rf3D3isHr7f0/view?usp=sharing",
    "Simple Linear Balanced": "https://drive.google.com/file/d/1x-UFJ1j0_Hb9Mi_gynmCsRg3esaITT2i/view?usp=drive_link",
    "Simple Linear Balanced (ECS only)": "https://drive.google.com/file/d/1mQ_xk9s4NtP8lVeY3pixhRs1Cl8uPun0/view?usp=drive_link",
    "Simple Linear Family": "https://drive.google.com/file/d/1hkBtuDy1NVZTeBkXKDz7r00pBA2jcWme/view?usp=drive_link",
    "Simple Linear Family (ECS only)": "https://drive.google.com/file/d/1D6AecBjGtztGtgDAbQEGybaxT_CziDXI/view?usp=drive_link",
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
    "Transformer + MLP": {
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
            "seq_len": 190,
        },
    },
    "Transformer + MLP (ECS only)": {
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
    "Transformer + MLP Non-MSA": {
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
            "seq_len": 190,
        },
    },
    "Transformer + MLP Non-MSA (ECS only)": {
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
    "Transformer + MLP ESM2": {
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
            "seq_len": 190,
        },
    },
    "Transformer + MLP ESM2 (ECS only)": {
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
    "Transformer + MLP Diverse": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on chunked train/val split from the single FASTA file. Chunks/Batches contained at-least one sequence from each claudin family with the remaining sequences filled in a round-robin manner, prioritising families with the most leftover sequences.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_single_fasta_diverse.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": {
            "embedding_dim": 768,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 190,
        },
    },
    "Transformer + MLP Diverse (ECS only)": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on chunked train/val split from the single FASTA file modified to contain only the ECS1/2 segments. Chunks/Batches contained at-least one sequence from each claudin family with the remaining sequences filled in a round-robin manner, prioritising families with the most leftover sequences.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_single_fasta_diverse_ecs_only.pt",
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
    "Transformer + MLP Balanced": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on chunked train/val split from the single FASTA file. Chunks/Batches contained equal number of sequences from each claudin family.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_single_fasta_balanced.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": {
            "embedding_dim": 768,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 190,
        },
    },
    "Transformer + MLP Balanced (ECS only)": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on chunked train/val split from the single FASTA file modified to contain only the ECS1/2 segments. Chunks/Batches contained equal number of sequences from each claudin family.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_single_fasta_balanced_ecs_only.pt",
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
    "Transformer + MLP Family": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on chunked train/val split from the single FASTA file. Chunks/Batches grouped by claudin family.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_single_fasta_family.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": {
            "embedding_dim": 768,
            "proj_dim": 128,
            "num_classes": 3,
            "num_heads": 4,
            "num_attention_blocks": 1,
            "dropout": 0.4,
            "seq_len": 190,
        },
    },
    "Transformer + MLP Family (ECS only)": {
        "class_name": "Transformer + MLP Classifier",
        "description": "Attention model with positional embeddings and fused pooled sequence features. Trained on chunked train/val split from the single FASTA file modified to contain only the ECS1/2 segments. Chunks/Batches grouped by claudin family.",
        "architecture": "Linear projection -> positional embedding -> self-attention blocks -> attention/mean/max pooling -> fusion MLP -> linear classifier",
        "uses_attention": True,
        "checkpoint_file": "transformer_mlp_classifier_single_fasta_family_ecs_only.pt",
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
    "Simple Linear": {
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
    "Simple Linear (ECS only)": {
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
    "Simple Linear Non-MSA": {
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
    "Simple Linear Non-MSA (ECS only)": {
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
    "Simple Linear ESM2": {
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
    "Simple Linear ESM2 (ECS only)": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on train/val split from the single FASTA file using ESM2 embeddings.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_esm2_ecs_only.pt",
        "compatible_embedder": "ESM2",
        "residue_slice": [(27, 81), (138, 164)],
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 640
        },
    },
    "Simple Linear Diverse": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on chunked train/val split from the single FASTA file. Chunks/Batches contained at-least one sequence from each claudin family with the remaining sequences filled in a round-robin manner, prioritisingfamilies with the most leftover sequences.",
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
    "Simple Linear Diverse (ECS only)": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on chunked train/val split from the single FASTA file modified to contain only the ECS1/2 segments. Chunks/Batches contained at-least one sequence from each claudin family with the remaining sequences filled in a round-robin manner, prioritisingfamilies with the most leftover sequences.",
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
    "Simple Linear Balanced": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on chunked train/val split from the single FASTA file. Chunks/Batches contained equal number of sequences from each claudin family.",
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
    "Simple Linear Balanced (ECS only)": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on chunked train/val split from the single FASTA file modified to contain only the ECS1/2 segments. Chunks/Batches contained equal number of sequences from each claudin family.",
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
    "Simple Linear Family": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on chunked train/val split from the single FASTA file. Chunks/Batches grouped by claudin family.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_single_fasta_family.pt",
        "compatible_embedder": "MSA Transformer",
        "kwargs": { 
            "n_classes": 3, 
            "dropout": 0.2,
            "embedding_dim": 768
        },
    },
    "Simple Linear Family (ECS only)": {
        "class_name": "Simple Linear Classifier",
        "description": "LayerNorm baseline using learned residue attention and a single linear head. Trained on chunked train/val split from the single FASTA file modified to contain only the ECS1/2 segments. Chunks/Batches grouped by claudin family.",
        "architecture": "LayerNorm -> linear attention scores -> softmax weights -> weighted sum -> dropout -> linear classifier",
        "uses_attention": False,
        "checkpoint_file": "simple_linear_classifier_single_fasta_family_ecs_only.pt",
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