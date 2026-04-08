from argparse import Namespace
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
import torch

from core.config import CHECKPOINTS_DIR

try:
    import esm
except Exception:
    esm = None


EMBEDDER_MODEL_NAME = "esm_msa1b_t12_100M_UR50S"
ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_ALLOWED_CHARS = set("ACDEFGHIKLMNPQRSTVWY")


def _embedder_checkpoint_paths(model_name: str) -> tuple[Path, Path]:
    checkpoint_stem = f"{model_name}_state_dict"
    return (
        CHECKPOINTS_DIR / f"{checkpoint_stem}.pt",
        CHECKPOINTS_DIR / f"{checkpoint_stem}.alphabet",
    )


def _msa_model_args_from_state_dict(state_dict: dict[str, torch.Tensor], alphabet) -> Namespace:
    embed_dim = state_dict["embed_tokens.weight"].shape[1]
    layers = len({int(key.split(".")[1]) for key in state_dict if key.startswith("layers.")})
    ffn_embed_dim = state_dict["layers.0.feed_forward_layer.layer.fc1.weight"].shape[0]
    max_positions = state_dict["embed_positions.weight"].shape[0] - alphabet.padding_idx - 1
    embed_positions_msa_dim = state_dict["msa_position_embedding"].shape[-1]
    attention_heads = max(1, embed_dim // 64)

    return Namespace(
        arch="msa_transformer",
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_embed_dim,
        attention_heads=attention_heads,
        layers=layers,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        max_tokens=2**14,
        max_tokens_per_msa=2**14,
        max_positions=max_positions,
        embed_positions_msa=True,
        embed_positions_msa_dim=embed_positions_msa_dim,
    )


def _load_embedder_from_checkpoints(model_name: str):
    state_path, alphabet_path = _embedder_checkpoint_paths(model_name)
    if not state_path.exists() or not alphabet_path.exists():
        raise FileNotFoundError(f"Missing embedder checkpoint files for {model_name} in {CHECKPOINTS_DIR}")

    state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
    alphabet = torch.load(alphabet_path, map_location="cpu", weights_only=False)
    model_data = {
        "args": _msa_model_args_from_state_dict(state_dict, alphabet),
        "model": state_dict,
    }
    model, _ = esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data=None)
    return model, alphabet


def _download_and_cache_embedder(model_name: str):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    state_path, alphabet_path = _embedder_checkpoint_paths(model_name)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_path)
    torch.save(alphabet, alphabet_path)
    return model, alphabet


def clean_sequence_for_esmfold(sequence: str) -> str:
    cleaned = []
    for char in str(sequence).upper():
        if char in ESMFOLD_ALLOWED_CHARS:
            cleaned.append(char)
    return "".join(cleaned)


class MSAEmbedder:
    def __init__(self, model_name=EMBEDDER_MODEL_NAME, device=None):
        if esm is None:
            raise ImportError("fair-esm is not installed. Install it from requirements.txt.")
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        try:
            self.model, self.alphabet = _load_embedder_from_checkpoints(model_name)
            print(f"[EMBED] Loaded embedder from checkpoints model={model_name}")
        except Exception:
            self.model, self.alphabet = _download_and_cache_embedder(model_name)
            print(f"[EMBED] Downloaded embedder model={model_name}")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)
        self.valid_chars = set(self.alphabet.all_toks)
        self.model.eval()

    def _clean_sequences(self, sequences):
        cleaned = []
        for seq in sequences:
            cleaned.append("".join([c if c in self.valid_chars else "-" for c in str(seq).upper()]))
        return cleaned

    @staticmethod
    def pad_or_truncate(sequences, seq_length, pad_char='-'):
        processed = []
        for seq in sequences:
            processed.append(seq[:seq_length] if len(seq) > seq_length else seq.ljust(seq_length, pad_char))
        return processed

    def embed_sequences_per_residue(self, sequences, seq_length=190, batch_size=32, is_baseline=False):
        if is_baseline:
            print(f"[EMBED] Generating baseline embeddings seq_len={seq_length}")
        else:
            print(f"[EMBED] Start n_seq={len(sequences)} seq_len={seq_length} batch={batch_size}")
        sequences = self._clean_sequences(sequences)
        sequences = self.pad_or_truncate(sequences, seq_length) if seq_length is not None else sequences
        all_embeddings = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(sequences))
            batch = sequences[start:end]
            msa_batch = [(f"seq{start + j}", seq) for j, seq in enumerate(batch)]
            _, _, batch_tokens = self.batch_converter(msa_batch)
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[12], return_contacts=False)
            token_embeddings = results["representations"][12][:, :, 1:, :]
            all_embeddings.append(token_embeddings.cpu())
        embeddings = torch.cat(all_embeddings, dim=1).squeeze()
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        if is_baseline:
            print(f"[EMBED] Done Baseline embeddings shape={tuple(embeddings.shape)}")
        else:
            print(f"[EMBED] Done shape={tuple(embeddings.shape)}")
        return embeddings


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = EMBEDDER_MODEL_NAME):
    return MSAEmbedder(model_name=model_name)


def build_baseline_embeddings(embedder: MSAEmbedder, seq_len: int) -> torch.Tensor:
    """Create baseline embeddings using zero-padded sequence representation.
    The baseline represents "no information" for Integrated Gradients attribution.
    """
    # Create a padding-only sequence (all dashes), embed it to get baseline signal
    padding_seq = ["-" * seq_len]
    baseline_embedding = embedder.embed_sequences_per_residue(padding_seq, seq_length=seq_len, batch_size=1, is_baseline=True)
    # print(f"Baseline embedding shape: {baseline_embedding.shape}")
    return baseline_embedding
    # baseline = torch.full((1, 1, seq_len), embedder.alphabet.padding_idx, dtype=torch.long)
    # baseline_embedding = embedder.embed_sequences_per_residue(baseline, seq_length=seq_len, batch_size=1)
    # print(f"Padding index: {embedder.alphabet.padding_idx}")


def infer_structure_with_esmfold(sequence: str, out_dir: Path) -> Optional[Path]:
    cleaned_sequence = clean_sequence_for_esmfold(sequence)
    if not cleaned_sequence:
        return None
    try:
        request = Request(
            ESMFOLD_API_URL,
            data=cleaned_sequence.encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "text/plain; charset=utf-8",
                "Accept": "text/plain",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urlopen(request, timeout=300) as response:
            pdb_string = response.read().decode("utf-8")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "prediction.pdb"
        path.write_text(pdb_string)
        return path
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None
