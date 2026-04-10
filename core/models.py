from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

from core.config import CHECKPOINTS_DIR, MODEL_REGISTRY, resolve_checkpoint_url


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class SingleSequenceAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = ResidualMLPBlock(dim, hidden_dim=dim * 4, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x = self.ffn(x)
        return x


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, R, D)
        h = self.norm(x)
        logits = self.score(h).squeeze(-1) # (B, R)
        attn = torch.softmax(logits, dim=-1) # (B, R) — residue attention weights
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1) # (B, D)
        return pooled, attn


class TransformerMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim=768,
        num_classes=3,
        num_heads=8,
        num_attention_blocks=2,
        dropout=0.1,
        max_residues=1024,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.pos_emb = nn.Embedding(max_residues, input_dim)
        self.emb_norm_before = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_blocks = nn.ModuleList(
            [
                SingleSequenceAttentionBlock(input_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_attention_blocks)
            ]
        )
        self.emb_norm_after = nn.LayerNorm(input_dim)
        self.residue_pool = AttentionPool(input_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(input_dim, hidden_dim=input_dim * 4, dropout=dropout),
            nn.LayerNorm(input_dim),
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x, return_attn=False):
        _, r, _ = x.shape
        device = x.device
        x = self.input_proj(x)
        res_ids = torch.arange(r, device=device)
        pos_emb = self.pos_emb(res_ids)[None, :, :]
        x = x + pos_emb
        x = self.emb_norm_before(x)
        x = self.dropout(x)
        for block in self.attention_blocks:
            x = block(x)
        x = self.emb_norm_after(x)
        pooled, residue_attn = self.residue_pool(x)
        mean_repr = x.mean(dim=1)
        max_repr = x.amax(dim=1)
        fused = torch.cat([pooled, mean_repr, max_repr], dim=-1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        if return_attn:
            return logits, residue_attn
        return logits


class SimpleLinearClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.norm = nn.LayerNorm(768)
        # attention scores per residue
        self.attn = nn.Linear(768, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, n_classes)

    def forward(self, x, mask=None):
        # x: (B, L, 768)
        x = self.norm(x)
        # attention weights over residues
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)
        # weighted sum -> (B, 768)
        seq_repr = torch.sum(x * weights.unsqueeze(-1), dim=1)
        seq_repr = self.dropout(seq_repr)
        logits = self.fc(seq_repr)
        return logits


class SimpleCNNClassifier(nn.Module):
    def __init__(
        self,
        n_classes=3,
        embedding_dim=768,
        n_filters=100,
        filter_sizes=None,
        dropout=0.1,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        # Normalization layer for input embeddings
        self.norm = nn.LayerNorm(embedding_dim)
        # Define multiple convolutional layers with different filter sizes.
        # Each filter looks at fs residues at a time across embedding width.
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )
        # Final fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Length, 768] input embeddings
            mask: [Batch, Length] bool tensor (True=data, False=padding)
        """
        # 1. Normalize and add channel dimension -> [Batch, 1, Length, 768]
        x = self.norm(x).unsqueeze(1)
        pooled_outputs = []

        for conv in self.convs:
            # 2. Apply convolution and ReLU -> [Batch, n_filters, L_out]
            conved = F.relu(conv(x)).squeeze(3)

            # 3. Apply masking logic aligned to convolution output length
            if mask is not None:
                # Get the filter size (height) of the current convolution
                fs = conv.kernel_size[0]
                # Because convolution reduces length by (fs - 1), align mask
                output_mask = mask[:, fs - 1 :].unsqueeze(1)
                # Fill padding positions with a very small value for max pooling
                conved = conved.masked_fill(~output_mask, -1e9)

            # 4. Global max pooling -> [Batch, n_filters]
            pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)

        # 5. Concatenate all features and apply dropout
        cat = self.dropout(torch.cat(pooled_outputs, dim=1))
        # 6. Final classification
        return self.fc(cat)


class SimpleTransformerClassifier(nn.Module):
    """Transformer-based classifier with positional encoding."""
    def __init__(self, embedding_dim=768, num_classes=3, num_heads=8, num_layers=2, dropout=0.1, max_seq_len=512):
        super().__init__()

        # Add positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.pool = AttentionPool(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_attn=False):
        # x: (batch_size, seq_len, embedding_dim)
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        # Add positional embeddings
        x = x + pos_emb
        transformer_out = self.transformer(x)
        
        pooled, residue_attn = self.pool(transformer_out)  
        logits = self.fc(pooled)
        
        if return_attn:
            return logits, residue_attn
        return logits


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        x = self.dropout(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, attn_dim=None, dropout=0.1):
        super().__init__()
        attn_dim = attn_dim or hidden_dim
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x):
        logits = self.score(x).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("br,brh->bh", weights, x)
        return pooled, weights


class ComplexTransformerClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        num_classes=3,
        max_len=512,
        hidden_dim=128,
        pos_emb_dim=128,
        num_conv_blocks=3,
        kernel_size=3,
        dropout=0.2,
        use_pc_features=False,
        pc_dim=3,
        pc_mlp_dim=32,
    ):
        super().__init__()
        self.use_pc_features = use_pc_features
        self.pc_dim = pc_dim if use_pc_features else 0

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(max_len, pos_emb_dim)
        self.pos_proj = nn.Linear(pos_emb_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout)
                for _ in range(num_conv_blocks)
            ]
        )

        self.pool = AttentionPooling(hidden_dim, attn_dim=hidden_dim, dropout=dropout)

        if self.use_pc_features:
            self.pc_mlp = nn.Sequential(
                nn.Linear(self.pc_dim, pc_mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pc_mlp_dim, pc_mlp_dim),
                nn.GELU(),
            )
            fusion_dim = hidden_dim + pc_mlp_dim
        else:
            self.pc_mlp = None
            fusion_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, pc_features=None, return_intermediate=False, return_attn=False):
        """
        x: (B, R, D)
        pc_features: (B, pc_dim) optional
        """
        b, r, _ = x.shape

        h = self.input_proj(x)
        pos = torch.arange(r, device=x.device).unsqueeze(0).expand(b, r)
        h = h + self.pos_proj(self.pos_emb(pos))

        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)

        pooled, attn_weights = self.pool(h)

        if self.use_pc_features:
            if pc_features is None:
                raise ValueError("pc_features must be provided when use_pc_features=True")
            pc_repr = self.pc_mlp(pc_features)
            fused = torch.cat([pooled, pc_repr], dim=-1)
        else:
            pc_repr = None
            fused = pooled

        logits = self.classifier(fused)

        if return_intermediate:
            return logits, attn_weights, h, pooled, pc_repr, fused
        if return_attn:
            return logits, attn_weights
        return logits


MODEL_CLASS_MAP = {
    "Transformer + MLP Classifier": TransformerMLPClassifier,
    "Simple Linear Classifier": SimpleLinearClassifier,
    "Simple CNN Classifier": SimpleCNNClassifier,
    "Transformer Classifier (simple)": SimpleTransformerClassifier,
    "Transformer Classifier (complex)": ComplexTransformerClassifier,
}


@dataclass
class LoadedModelBundle:
    model_name: str
    classifier: nn.Module
    uses_attention: bool
    description: str
    architecture: str


def _download_checkpoint_from_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp") as temp_file:
        temp_path = Path(temp_file.name)
    try:
        if "drive.google.com" in url:
            gdown = importlib.import_module("gdown")
            gdown.download(url=url, output=str(temp_path), quiet=False, fuzzy=True)
        else:
            raise RuntimeError("Only Google Drive checkpoint URLs are supported in this setup.")
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _checkpoint_filename_from_model_key(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"{slug}.pt"


def _ensure_checkpoint_file(model_name: str, checkpoint_file: str) -> Path:
    ckpt_path = CHECKPOINTS_DIR / _checkpoint_filename_from_model_key(model_name)
    if ckpt_path.exists():
        return ckpt_path

    # Reuse existing legacy filename if it is already present locally.
    legacy_path = CHECKPOINTS_DIR / checkpoint_file
    if legacy_path.exists():
        return legacy_path

    checkpoint_url = resolve_checkpoint_url(model_name=model_name, checkpoint_file=checkpoint_file)
    if not checkpoint_url:
        return ckpt_path

    st.toast(f"🚀 Downloading model: {model_name}")
    _download_checkpoint_from_url(checkpoint_url, ckpt_path)
    st.toast(f"🚀 Model ready: {model_name}")
    print(f"[MODEL] Downloaded checkpoint for model '{model_name}' to: {ckpt_path}")
    return ckpt_path


def load_classifier_bundle(model_name: str) -> LoadedModelBundle:
    cfg = MODEL_REGISTRY[model_name]
    print(f"[MODEL] Load model: {model_name}")

    classifier = MODEL_CLASS_MAP[cfg["class_name"]](**cfg.get("kwargs", {}))
    ckpt_path = _ensure_checkpoint_file(model_name=model_name, checkpoint_file=cfg["checkpoint_file"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint file {cfg['checkpoint_file']} in {CHECKPOINTS_DIR}. "
            "Set CHECKPOINT_GDRIVE_URLS_JSON with model-name -> Google Drive link mapping."
        )
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    classifier.load_state_dict(state, strict=False)
    classifier.eval()
    print(f"[MODEL] Ready: {model_name} ({ckpt_path})")
    return LoadedModelBundle(
        model_name=model_name,
        classifier=classifier,
        uses_attention=cfg["uses_attention"],
        description=cfg["description"],
        architecture=cfg["architecture"],
    )
