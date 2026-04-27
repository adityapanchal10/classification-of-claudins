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

    
    def embed_msa(self, sequences, seq_length=190, max_msa_depth=600):
        """
        Embed all sequences from ONE MSA file together (true MSA mode).
        Column attention operates across all sequences simultaneously.

        Args:
            sequences    : list of aligned sequences (all from the same MSA file)
            seq_length   : pad/truncate target length
            max_msa_depth: max sequences per forward pass (GPU memory limit)

        Returns:
            Tensor of shape (N, seq_length, 768)
        """
        sequences = self._clean_sequences(sequences)
        sequences = self.pad_or_truncate(sequences, seq_length) if seq_length is not None else sequences
        N = len(sequences)
        print(f"[EMBED] Start MSA Embedding n_seq={N} seq_len={seq_length}")

        all_embeddings = []

        for start in range(0, N, max_msa_depth):
            chunk = sequences[start: start + max_msa_depth]

            # Wrap all chunk sequences as a single MSA input
            msa_input = [(f'seq{start + j}', seq) for j, seq in enumerate(chunk)]

            # batch_converter: tokens shape → (1, depth, seq_len+1), +1 for BOS
            _, _, batch_tokens = self.batch_converter([msa_input])
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[12], return_contacts=False)

            # Extract representations: (1, depth, seq_len+1, 768)
            token_emb = results["representations"][12]
            token_emb = token_emb[:, :, 1:, :]    # Remove BOS → (1, depth, seq_len, 768)
            token_emb = token_emb.squeeze(0)       # → (depth, seq_len, 768)

            all_embeddings.append(token_emb.cpu())

        output_embeddings = torch.cat(all_embeddings, dim=0)
        assert len(output_embeddings.shape) == 3, f"Unexpected shape: {output_embeddings.shape}"
        print(f"[EMBED] Done shape={tuple(output_embeddings.shape)}")
        return output_embeddings  # (N, seq_len, 768)
    
    def embed_sequences_per_residue(self, sequences, seq_length=190, batch_size=32, is_baseline=False):
        if is_baseline:
            print(f"[EMBED] Generating baseline embeddings seq_len={seq_length}")
        else:
            print(f"[EMBED] Start n_seq={len(sequences)} seq_len={seq_length} batch={batch_size}")
        sequences = self._clean_sequences(sequences)
        sequences = self.pad_or_truncate(sequences, seq_length) if seq_length is not None else sequences
        all_embeddings = None
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
            token_embeddings = results["representations"][12]

            # Convert model output to shape (batch, residues, embed_dim).
            if token_embeddings.ndim == 4 and token_embeddings.shape[1] == 1:
                batch_embeddings = token_embeddings[:, 0, 1:, :]
            elif token_embeddings.ndim == 4 and token_embeddings.shape[0] == 1:
                batch_embeddings = token_embeddings[0, :, 1:, :]
            elif token_embeddings.ndim == 3:
                batch_embeddings = token_embeddings[:, 1:, :]
            else:
                raise RuntimeError(f"Unexpected embedding shape from model: {tuple(token_embeddings.shape)}")

            batch_embeddings = batch_embeddings.cpu()

            if all_embeddings is None:
                n_seq, n_res, embed_dim = len(sequences), batch_embeddings.shape[1], batch_embeddings.shape[2]
                all_embeddings = torch.empty((n_seq, n_res, embed_dim), dtype=batch_embeddings.dtype)

            all_embeddings[start:end] = batch_embeddings

        embeddings = all_embeddings
        if is_baseline:
            print(f"[EMBED] Done Baseline embeddings shape={tuple(embeddings.shape)}")
        else:
            print(f"[EMBED] Done shape={tuple(embeddings.shape)}")
        return embeddings


def get_embedder(model_name: str = EMBEDDER_MODEL_NAME) -> MSAEmbedder:
    """Return a session-scoped embedder singleton.

    The heavy ESM model is kept in ``st.session_state`` so it survives across
    Streamlit reruns without being re-loaded from disk every time.  If the
    requested *model_name* differs from the cached one the old instance is
    released first so memory is freed before the new one is allocated.
    """
    cached: MSAEmbedder | None = st.session_state.get("_embedder_instance")
    if cached is not None and cached.model_name == model_name:
        return cached

    # Release previous instance before allocating a new one.
    if cached is not None:
        st.session_state.pop("_embedder_instance", None)
        st.session_state.pop("_baseline_cache", None)
        del cached
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[EMBED] Released previous embedder")

    embedder = MSAEmbedder(model_name=model_name)
    st.session_state["_embedder_instance"] = embedder
    return embedder


def build_baseline_embeddings(seq_len: int, embedding_dim: int = 768) -> torch.Tensor:
    """Create baseline embeddings using zero values.
    The baseline represents "no information" for Integrated Gradients attribution.

    Results are cached per *seq_len* in session state so the (expensive)
    embedding call only happens once per sequence length.
    """
    cache: dict = st.session_state.get("_baseline_cache", {})
    if seq_len in cache:
        print(f"[EMBED] Baseline cache hit seq_len={seq_len}")
        return cache[seq_len]

    baseline_embedding = torch.zeros(1, seq_len, embedding_dim)
    cache[seq_len] = baseline_embedding
    st.session_state["_baseline_cache"] = cache
    return baseline_embedding


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
