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

_MAX_CACHED_CLASSIFIERS = 2


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.4):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.norm    = nn.LayerNorm(dim)
        self.fc1     = nn.Linear(dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class SingleSequenceAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.4):
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn     = ResidualMLPBlock(dim, hidden_dim=dim * 2, dropout=dropout)

    def forward(self, x):
        x_norm   = self.norm(x)
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
        embedding_dim=768,          # ESM embedding dim — fixed
        proj_dim=128,
        num_classes=3,
        num_heads=4,
        num_attention_blocks=1,
        dropout=0.4,
        seq_len=220,
    ):
        super().__init__()

        # ── Stage 1: Project ESM embeddings down to a manageable size ─────────
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),   # 768 → 128
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout)
        )

        # Stage 2: Positional embeddings (now 128-dim, not 768/640-dim)
        self.pos_emb = nn.Embedding(seq_len, proj_dim)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        self.emb_norm_before = nn.LayerNorm(proj_dim)
        self.dropout         = nn.Dropout(dropout)

        # ── Stage 3: Single attention block  ───────────────────────────
        self.attention_blocks = nn.ModuleList([
            SingleSequenceAttentionBlock(
                dim=proj_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_attention_blocks)
        ])

        self.emb_norm_after = nn.LayerNorm(proj_dim)

        # ── Stage 4: Attention pooling ────────────────────────────────────────
        self.residue_pool = AttentionPool(proj_dim)

        # ── Stage 5: Lightweight classifier head ──────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim // 2),   # 128 → 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, proj_dim // 4),  # 64 → 32
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 4, num_classes)  # 32 → 3
        )

    def forward(self, x, return_attn=False, return_pooled=False):
        """
        x: (B, R, D) — batch, residues, ESM embedding dim (768)
        """
        B, R, D = x.shape
        device  = x.device

        # Truncate to the maximum positional embedding length when needed.
        max_len = self.pos_emb.num_embeddings
        if R > max_len:
            x = x[:, :max_len, :]
            R = max_len

        # Project 768 → 128
        x = self.input_proj(x)                          # (B, R, 128)

        # Add positional embeddings
        res_ids = torch.arange(R, device=device)
        pos_emb = self.pos_emb(res_ids)[None, :, :]     # (1, R, 128)
        x = x + pos_emb

        # Pre-norm + dropout
        x = self.emb_norm_before(x)
        x = self.dropout(x)

        # Attention block(s)
        for block in self.attention_blocks:
            x = block(x)

        x = self.emb_norm_after(x)                      # (B, R, 128)

        # Attention pooling → single vector per sequence
        pooled, residue_attn = self.residue_pool(x)     # (B, 128)

        # Classify
        class_logits = self.head(pooled)                # (B, 3)

        outputs = [class_logits]

        if return_pooled:
            outputs.append(pooled)
        if return_attn:
            outputs.append(residue_attn)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]


class SimpleLinearClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout=0.2, embedding_dim=768):
        super().__init__()
        self.n_classes = n_classes
        self.norm = nn.LayerNorm(embedding_dim)
        # attention scores per residue
        self.attn = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, n_classes)

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

class SelectiveStateSpaceBlock(nn.Module):
    """
    Selective state-space layer (Mamba-inspired).
    Pure PyTorch implementation for portability and training stability.
    
    Input: (B, L, D) where D is model dimension
    Output: (B, L, D)
    """
    
    def __init__(self, dim, state_dim=64, expand_factor=2, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        inner_dim = dim * expand_factor
        
        # Projection to hidden dimension
        self.proj_in = nn.Linear(dim, inner_dim * 2)  # hidden + gate
        
        # State-space parameters (learnable)
        self.A_log = nn.Parameter(torch.randn(inner_dim, state_dim) * 0.1)
        self.B = nn.Linear(dim, state_dim)
        self.C = nn.Linear(dim, state_dim)
        self.D = nn.Parameter(torch.ones(inner_dim))
        
        # Gate for selective mechanism
        self.gate_proj = nn.Linear(dim, inner_dim)
        
        # Output
        self.proj_out = nn.Linear(inner_dim, dim)
        
        # Normalization and residual
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def selective_scan(self, u, A, B, C, D):
        """
        Reference selective scan implementation (not currently used in forward).
        Kept for future optimization or bidirectional variants.
        
        Args:
            u: (B, L, dim) input
            A: (dim, state_dim) state matrix
            B: (B, L, state_dim) input projection (selective)
            C: (B, L, state_dim) output projection (selective)
            D: (dim,) feedthrough parameter
            
        Returns:
            output: (B, L, dim)
        """
        B_batch, L, dim = u.shape
        state_dim = A.shape[1]
        device = u.device
        
        # Discretize A: A_d = exp(dt * A) converts continuous to discrete time
        dt = 0.1
        A_d = torch.exp(dt * A)  # (dim, state_dim)
        
        # Initialize state
        h = torch.zeros(B_batch, dim, state_dim, device=device, dtype=u.dtype)
        
        outputs = []
        for t in range(L):
            u_t = u[:, t, :]  # (B, dim)
            B_t = B[:, t, :]  # (B, state_dim)
            C_t = C[:, t, :]  # (B, state_dim)
            
            # Update state: h = A_d * h + B_t @ u_t
            # Reshape for matrix mult: h is (B, dim, state_dim)
            h = torch.einsum('ds,bs->bds', A_d, h.reshape(B_batch, dim, state_dim).squeeze(-1)) + torch.outer(u_t.squeeze(0), B_t.squeeze(0)).unsqueeze(0)
            
            # Simpler recurrence: h_new = A_d * h + B_t * u_t (expanded)
            h = (h * A_d.unsqueeze(0)) + torch.einsum('bs,bd->bds', B_t, u_t)
            
            # Output: y = C_t @ h + D * u_t
            y = torch.einsum('bds,bs->bd', h, C_t) + D.unsqueeze(0) * u_t
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # (B, L, dim)
    
    def forward(self, x):
        """
        Selective state-space block forward pass.
        
        Flow: normalize → project to hidden+gate → apply gating → SSM recurrence → project out
        
        Args:
            x: (B, L, dim) input tensor
        
        Returns:
            (B, L, dim) output tensor with residual connection
        """
        residual = x
        x = self.norm(x)  # Pre-normalization
        
        B, L, D = x.shape
        
        # === Expand and Gate ===
        # Project input to hidden dimension (dim → 2*inner_dim: [hidden, gate])
        proj = self.proj_in(x)  # (B, L, 2 * inner_dim)
        inner_dim = self.dim * self.expand_factor
        hidden, gate = proj.chunk(2, dim=-1)  # each (B, L, inner_dim)
        
        # Apply gating: sigmoid controls how much hidden signal passes through
        gate = torch.sigmoid(gate)
        hidden = hidden * gate
        
        # === Generate selective parameters ===
        # B, C control input and output projection per timestep (selective mechanism)
        B_proj = self.B(x)  # (B, L, state_dim) - input selectivity
        C_proj = self.C(x)  # (B, L, state_dim) - output selectivity
        
        # === Selective SSM Recurrence ===
        # Maintain a hidden state that recurrently accumulates information
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)  # (B, state_dim)
        outputs = []
        
        for t in range(L):
            # Aggregate hidden features: compress inner_dim → scalar signal
            hidden_signal = hidden[:, t, :].mean(dim=-1, keepdim=True)  # (B, 1)
            
            # Update state: decay old state + modulate with new input
            h = h * 0.95 + B_proj[:, t, :] * hidden_signal  # (B, state_dim)
            
            # Compute output: pass state through nonlinearity, project back to inner_dim
            state_out = torch.tanh(h)  # (B, state_dim)
            state_contrib = (state_out.mean(dim=-1, keepdim=True) * self.D).expand(B, inner_dim)
            
            # Combine hidden features with state contribution (residual-like)
            y_t = hidden[:, t, :] + state_contrib  # (B, inner_dim)
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # (B, L, inner_dim)
        
        # === Project back to original dimension ===
        output = self.proj_out(output)  # (B, L, D)
        output = self.dropout(output)
        
        # Residual connection
        return residual + output


class AttentionPoolMamba(nn.Module):
    """
    Attention-based pooling: learns which residues matter most.
    
    For each position in the sequence, compute a learned importance score.
    Compute attention weights via softmax, then weighted average across positions.
    
    Example: if residues 28-81 (extracellular loop) are important for classification,
    attention will learn to assign them higher weights.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)  # Learn importance per position

    def forward(self, x):
        """
        Args:
            x: (B, L, D) sequence of residue embeddings
        
        Returns:
            pooled: (B, D) weighted average across L positions
            attn: (B, L) learned attention weights (inspect for interpretability)
        """
        h = self.norm(x)  # Normalize before scoring
        logits = self.score(h).squeeze(-1)  # (B, L) score per position
        attn = torch.softmax(logits, dim=-1)  # (B, L) normalized weights
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # (B, D) weighted sum
        return pooled, attn


class Mamba2ESMClassifier(nn.Module):
    """
    Medium-sized Mamba-2-inspired classifier on cached ESM2 embeddings.
    
    Architecture:
    - Input projection: 640 (ESM2) → 256 (internal dim)
    - 6× selective SSM blocks
    - Attention or mean pooling
    - Classification head: 256 → 128 → 3
    
    Args:
        embedding_dim: input dimension (640 for ESM2, 768 for MSA Transformer)
        model_dim: internal model dimension (default 256)
        state_dim: state space dimension (default 64)
        expand_factor: expansion factor for SSM (default 2)
        num_ssm_blocks: number of SSM blocks (default 6)
        dropout: dropout rate (default 0.2)
        num_classes: number of output classes (default 3)
        pooling_mode: 'attention' or 'mean' (default 'attention')
    """
    
    def __init__(
        self,
        embedding_dim=640,
        model_dim=256,
        state_dim=64,
        expand_factor=2,
        num_ssm_blocks=6,
        dropout=0.2,
        num_classes=3,
        pooling_mode='attention',
    ):
        super().__init__()
        
        self.uses_attn = (pooling_mode == 'attention')
        self.pooling_mode = pooling_mode
        self.model_dim = model_dim
        
        # Input projection: ESM2 (640) → model_dim (256)
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        
        # SSM blocks
        self.ssm_blocks = nn.ModuleList([
            SelectiveStateSpaceBlock(
                dim=model_dim,
                state_dim=state_dim,
                expand_factor=expand_factor,
                dropout=dropout,
            )
            for _ in range(num_ssm_blocks)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(model_dim)
        
        # Pooling
        if pooling_mode == 'attention':
            self.pool = AttentionPoolMamba(model_dim)
            pool_dim = model_dim
        elif pooling_mode == 'mean':
            self.pool = None
            pool_dim = model_dim
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, model_dim // 2),  # 256 → 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, model_dim // 4),  # 128 → 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 4, num_classes),  # 64 → 3
        )
    
    def forward(self, x, return_attn=False, return_pooled=False):
        """
        Full Mamba-2 ESM classifier forward pass.
        
        Flow: input_proj → SSM blocks → pooling → classification head
        
        Args:
            x: (B, L, embedding_dim) ESM2 embeddings (640-dim, 190 residues)
            return_attn: if True, also return attention weights (for interpretability)
            return_pooled: if True, also return pooled representation before head
        
        Returns:
            logits: (B, 3) class predictions
            pooled: (B, 256) [optional] aggregated sequence representation
            attn: (B, L) [optional] attention weights per position
        """
        # === Process sequence through SSM blocks ===
        x = self.input_proj(x)  # (B, L, 640) → (B, L, 256)
        
        # Stack 6 SSM blocks for recurrent-style processing
        for block in self.ssm_blocks:
            x = block(x)  # (B, L, 256)
        
        x = self.final_norm(x)  # Final layer normalization
        
        # === Pooling: compress (B, L, 256) → (B, 256) ===
        # Attention pooling: learn which residues matter
        # Mean pooling: uniform average (baseline/ablation)
        if self.pooling_mode == 'attention':
            pooled, attn_weights = self.pool(x)  # (B, 256), (B, L)
        else:  # mean pooling
            pooled = x.mean(dim=1)  # (B, 256)
            attn_weights = None
        
        # === Classification head ===
        class_logits = self.head(pooled)  # (B, 256) → (B, 3)
        
        # === Prepare outputs ===
        outputs = [class_logits]
        
        if return_pooled:
            outputs.append(pooled)
        if return_attn:
            if attn_weights is not None:
                outputs.append(attn_weights)
            else:
                # Mean pooling: return uniform attention as placeholder
                B, L = x.shape[0], x.shape[1]
                outputs.append(torch.ones(B, L, device=x.device) / L)
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]



MODEL_CLASS_MAP = {
    "Transformer + MLP Classifier": TransformerMLPClassifier,
    "Simple Linear Classifier": SimpleLinearClassifier,
    "Simple CNN Classifier": SimpleCNNClassifier,
    "Transformer Classifier (simple)": SimpleTransformerClassifier,
    "Transformer Classifier (complex)": ComplexTransformerClassifier,
    "Mamba2 Classifier": Mamba2ESMClassifier,
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
            gdown.download(url=url, output=str(temp_path), quiet=False)
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
    ckpt_name = checkpoint_file or _checkpoint_filename_from_model_key(model_name)
    ckpt_path = CHECKPOINTS_DIR / ckpt_name
    if ckpt_path.exists():
        return ckpt_path

    checkpoint_url = resolve_checkpoint_url(model_name=model_name, checkpoint_file=checkpoint_file)
    if not checkpoint_url:
        return ckpt_path

    st.toast(f"🚀 Downloading model: {model_name}")
    _download_checkpoint_from_url(checkpoint_url, ckpt_path)
    st.toast(f"🚀 Model ready: {model_name}")
    print(f"[MODEL] Downloaded checkpoint for model '{model_name}' to: {ckpt_path}")
    return ckpt_path


def _load_classifier_bundle_from_disk(model_name: str) -> LoadedModelBundle:
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

    # Validate that the checkpoint keys actually match the model.
    result = classifier.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"[MODEL] WARNING {model_name}: missing keys in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[MODEL] WARNING {model_name}: unexpected keys in checkpoint: {result.unexpected_keys}")

    classifier.eval()
    print(f"[MODEL] Ready: {model_name} params={sum(p.numel() for p in classifier.parameters())} ({ckpt_path.name})")
    return LoadedModelBundle(
        model_name=model_name,
        classifier=classifier,
        uses_attention=cfg["uses_attention"],
        description=cfg["description"],
        architecture=cfg["architecture"],
    )


def load_classifier_bundle(model_name: str) -> LoadedModelBundle:
    """Return a classifier bundle, reusing the session-cached instance when possible.

    Classifiers are lightweight (~1-5 MB) so we keep up to
    ``_MAX_CACHED_CLASSIFIERS`` in ``st.session_state`` (enough for the
    Compare page which needs two at once).  When the limit is exceeded the
    oldest entries are evicted first.
    """
    cache: dict[str, LoadedModelBundle] = st.session_state.get("_classifier_cache", {})
    if model_name in cache:
        print(f"[MODEL] Cache hit: {model_name}")
        return cache[model_name]

    # Evict oldest entries when the cache is full.
    while len(cache) >= _MAX_CACHED_CLASSIFIERS:
        oldest = next(iter(cache))
        cache.pop(oldest)
        print(f"[MODEL] Evicted cached bundle: {oldest}")

    import gc; gc.collect()
    bundle = _load_classifier_bundle_from_disk(model_name)
    cache[model_name] = bundle
    st.session_state["_classifier_cache"] = cache
    return bundle
