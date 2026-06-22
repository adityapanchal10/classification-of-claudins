from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import shutil
import subprocess
import tempfile

import streamlit as st
import torch

from core.config import CHECKPOINTS_DIR

try:
    import esm
except Exception:
    esm = None

def _load_fasta_sequences(fasta_path) -> list[str]:
    """Parse a FASTA file and return a list of sequences (headers stripped)."""
    sequences = []
    current_seq: list[str] = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
    if current_seq:
        sequences.append("".join(current_seq))
    return sequences


def _mafft_available() -> bool:
    return shutil.which("mafft") is not None


def _align_to_reference_msa(query_sequences: list[str], reference_msa_path) -> tuple[list[str], bool]:
    """
    Align query sequences into an existing reference MSA using mafft --add.

    Uses ``mafft --add <queries> --keeplength <reference>`` so the reference
    alignment columns are preserved and query sequences are inserted without
    disturbing them.

    Returns:
        (aligned_query_seqs, success)
        - aligned_query_seqs: query sequences with gap characters inserted to
          match the reference alignment width.  Falls back to the original
          (unaligned) sequences if mafft is unavailable or fails.
        - success: True when mafft ran successfully, False on fallback.
    """
    if not _mafft_available():
        print("[EMBED] mafft not found — skipping alignment, using raw sequences")
        return query_sequences, False

    n_queries = len(query_sequences)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        query_fasta = tmp / "queries.fasta"
        output_fasta = tmp / "aligned.fasta"

        # Write query sequences to a temporary FASTA
        with open(query_fasta, "w") as fh:
            for i, seq in enumerate(query_sequences):
                fh.write(f">query{i}\n{seq}\n")

        # Run mafft: add queries into the reference MSA, keep reference column width
        cmd = [
            "mafft",
            "--add", str(query_fasta),
            "--keeplength",
            "--quiet",
            str(reference_msa_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            print("[EMBED] mafft timed out — falling back to raw sequences")
            return query_sequences, False
        except Exception as e:
            print(f"[EMBED] mafft error: {e} — falling back to raw sequences")
            return query_sequences, False

        if result.returncode != 0:
            print(f"[EMBED] mafft non-zero exit ({result.returncode}) — falling back to raw sequences")
            print(f"[EMBED] mafft stderr: {result.stderr[:500]}")
            return query_sequences, False

        # mafft --add outputs the full alignment (reference rows + query rows).
        # Parse all sequences and return only the last n_queries entries.
        output_fasta.write_text(result.stdout)
        all_aligned = _load_fasta_sequences(output_fasta)

        if len(all_aligned) < n_queries:
            print("[EMBED] mafft output fewer sequences than expected — falling back")
            return query_sequences, False

        aligned_queries = all_aligned[-n_queries:]
        n_ref_rows = len(all_aligned) - n_queries
        print(f"[EMBED] mafft alignment done n_queries={n_queries} n_ref={n_ref_rows} aligned_len={len(aligned_queries[0]) if aligned_queries else 0}")
        return aligned_queries, True


EMBEDDER_SPECS = {
    "MSA Transformer": {
        "model_name": "esm_msa1b_t12_100M_UR50S",
        "final_layer": 12,
        "embedding_dim": 768,
        "supports_msa_mode": True,
    },
    "ESM2": {
        "model_name": "esm2_t30_150M_UR50D",
        "final_layer": 30,
        "embedding_dim": 640,
        "supports_msa_mode": False,
    },
}

DEFAULT_EMBEDDER_NAME = "MSA Transformer"
EMBEDDER_MODEL_NAME = EMBEDDER_SPECS[DEFAULT_EMBEDDER_NAME]["model_name"]
ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_ALLOWED_CHARS = set("ACDEFGHIKLMNPQRSTVWY")


def resolve_embedder_spec(model_name: str | None = None) -> dict[str, object]:
    requested = (model_name or DEFAULT_EMBEDDER_NAME).strip()
    if requested in EMBEDDER_SPECS:
        return EMBEDDER_SPECS[requested]
    for spec in EMBEDDER_SPECS.values():
        if spec["model_name"] == requested:
            return spec
    return EMBEDDER_SPECS[DEFAULT_EMBEDDER_NAME]


def available_embedder_names() -> list[str]:
    return list(EMBEDDER_SPECS.keys())


def embedder_supports_msa_mode(model_name: str | None = None) -> bool:
    return bool(resolve_embedder_spec(model_name)["supports_msa_mode"])


def _embedder_checkpoint_path(model_name: str) -> Path:
    return CHECKPOINTS_DIR / f"{model_name}_checkpoint.pt"


def _load_embedder_from_checkpoints(model_name: str):
    state_path = _embedder_checkpoint_path(model_name)
    if not state_path.exists():
        raise FileNotFoundError(f"Missing embedder checkpoint file for {model_name} in {CHECKPOINTS_DIR}")

    model, alphabet = torch.load(state_path, map_location="cpu", weights_only=False)
    return model, alphabet


def _download_and_cache_embedder(model_name: str):
    state_path = _embedder_checkpoint_path(model_name)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    torch.save((model, alphabet), state_path)
    return model, alphabet


def clean_sequence_for_esmfold(sequence: str) -> str:
    cleaned = []
    for char in str(sequence).upper():
        if char in ESMFOLD_ALLOWED_CHARS:
            cleaned.append(char)
    return "".join(cleaned)


class ESMEmbedder:
    def __init__(self, model_name=EMBEDDER_MODEL_NAME, device=None):
        if esm is None:
            raise ImportError("fair-esm is not installed. Install it from requirements.txt.")
        spec = resolve_embedder_spec(model_name)
        self.display_name = next(name for name, item in EMBEDDER_SPECS.items() if item["model_name"] == spec["model_name"])
        self.model_name = str(spec["model_name"])
        self.final_layer = int(spec["final_layer"])
        self.embedding_dim = int(spec["embedding_dim"])
        self.supports_msa_mode = bool(spec["supports_msa_mode"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        try:
            self.model, self.alphabet = _load_embedder_from_checkpoints(self.model_name)
            print(f"[EMBED] Loaded embedder from checkpoints model={self.model_name}")
        except Exception:
            self.model, self.alphabet = _download_and_cache_embedder(self.model_name)
            print(f"[EMBED] Downloaded embedder model={self.model_name}")
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

    
    def embed_msa(self, sequences, seq_length=190, max_msa_depth=600, reference_msa_path=None):
        """
        Embed all sequences from ONE MSA file together (true MSA mode).
        Column attention operates across all sequences simultaneously.

        Args:
            sequences         : list of aligned sequences (all from the same MSA file)
            seq_length        : pad/truncate target length
            max_msa_depth     : max sequences per forward pass (GPU memory limit)
            reference_msa_path: optional path to a FASTA reference MSA.  When
                                provided, query sequences are first aligned into
                                the reference MSA using ``mafft --add --keeplength``
                                (if mafft is available), then reference rows are
                                prepended as MSA context.  Only the query rows are
                                returned in the output tensor.

        Returns:
            Tensor of shape (N, seq_length, embedding_dim)
        """
        if not self.supports_msa_mode:
            return self.embed_sequences_per_residue(
                sequences,
                seq_length=seq_length,
                batch_size=len(sequences),
            )

        # Align queries into the reference MSA when requested
        ref_sequences: list[str] = []
        if reference_msa_path is not None:
            aligned_queries, mafft_ok = _align_to_reference_msa(sequences, reference_msa_path)
            if not mafft_ok:
                st.warning(
                    "⚠️ **mafft not found** — sequences could not be aligned to the reference MSA. "
                    "Embedding will proceed without alignment. Install mafft and restart the app for proper MSA context.",
                    icon="⚠️",
                )
            sequences = aligned_queries

            # Load reference rows to prepend as MSA context.
            # The reference MSA defines the canonical alignment width — all
            # sequences (reference and query alike) are forced to that length.
            ref_sequences = _load_fasta_sequences(reference_msa_path)
            ref_sequences = self._clean_sequences(ref_sequences)
            ref_width = len(ref_sequences[0]) if ref_sequences else seq_length
            ref_sequences = self.pad_or_truncate(ref_sequences, ref_width)
            # Override seq_length so query sequences are padded/truncated to the
            # same column count as the reference alignment, regardless of mafft.
            seq_length = ref_width
            print(f"[EMBED] Reference MSA path={reference_msa_path} n_ref={len(ref_sequences)} ref_width={ref_width} mafft={mafft_ok}")

        sequences = self._clean_sequences(sequences)
        sequences = self.pad_or_truncate(sequences, seq_length) if seq_length is not None else sequences
        N = len(sequences)
        print(f"[EMBED] Start {self.model_name} Embedding n_seq={N} seq_len={seq_length} msa_mode=True ref_msa={reference_msa_path is not None}")

        all_embeddings = []

        # Build a single MSA input: reference rows first (context), then all query sequences
        ref_rows = [(f'ref{i}', seq) for i, seq in enumerate(ref_sequences)]
        query_rows = [(f'seq{j}', seq) for j, seq in enumerate(sequences)]
        msa_input = ref_rows + query_rows

        # batch_converter: tokens shape → (1, depth, seq_len+1), +1 for BOS
        _, _, batch_tokens = self.batch_converter([msa_input])
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.final_layer], return_contacts=False)

        # Extract representations: (1, depth, seq_len+1, embedding_dim)
        token_emb = results["representations"][self.final_layer]
        token_emb = token_emb[:, :, 1:, :]    # Remove BOS → (1, depth, seq_len, embedding_dim)
        token_emb = token_emb.squeeze(0)       # → (depth, seq_len, embedding_dim)

        # Keep only the query rows (skip the prepended reference rows)
        n_ref = len(ref_sequences)
        all_embeddings.append(token_emb[n_ref:, :, :].cpu())

        output_embeddings = torch.cat(all_embeddings, dim=0)
        assert len(output_embeddings.shape) == 3, f"Unexpected shape: {output_embeddings.shape}"
        print(f"[EMBED] Done shape={tuple(output_embeddings.shape)}")
        return output_embeddings  # (N, seq_len, embedding_dim)
    
    def embed_sequences_per_residue(self, sequences, seq_length=190, batch_size=1, is_baseline=False):
        sequences = self._clean_sequences(sequences)
        sequences = self.pad_or_truncate(sequences, seq_length) if seq_length is not None else sequences
        N = len(sequences)

        if is_baseline:
            print(f"[EMBED] Generating baseline embeddings seq_len={seq_length}")
        else:
            print(f"[EMBED] Start {self.model_name} Embedding n_seq={N} seq_len={seq_length} msa_mode=False")
            
        all_embeddings = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(sequences))
            batch = sequences[start:end]

            if self.supports_msa_mode:
                msa_inputs = [[(f"seq{start + i}", seq)] for i, seq in enumerate(batch)]
                _, _, batch_tokens = self.batch_converter(msa_inputs)
            else:
                batch_inputs = [(f"seq{start + i}", seq) for i, seq in enumerate(batch)]
                _, _, batch_tokens = self.batch_converter(batch_inputs)
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.final_layer], return_contacts=False)
            token_emb = results["representations"][self.final_layer]
            if self.supports_msa_mode:
                token_emb = token_emb[:, 0, 1:, :]
            else:
                # ESM2 adds both BOS and EOS tokens; remove both.
                token_emb = token_emb[:, 1:-1, :]
            all_embeddings.append(token_emb.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        if is_baseline:
            print(f"[EMBED] Done Baseline embeddings shape={tuple(embeddings.shape)}")
        else:
            print(f"[EMBED] Done shape={tuple(embeddings.shape)}")
        return embeddings


def get_embedder(model_name: str | None = None) -> ESMEmbedder:
    """Return a session-scoped embedder singleton.

    The heavy ESM model is kept in ``st.session_state`` so it survives across
    Streamlit reruns without being re-loaded from disk every time.  If the
    requested *model_name* differs from the cached one the old instance is
    released first so memory is freed before the new one is allocated.
    """
    if model_name is None:
        model_name = st.session_state.get("global_embedder_name", DEFAULT_EMBEDDER_NAME)

    spec = resolve_embedder_spec(model_name)
    requested_model_name = str(spec["model_name"])
    cached: ESMEmbedder | None = st.session_state.get("_embedder_instance")
    if cached is not None and cached.model_name == requested_model_name:
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

    embedder = ESMEmbedder(model_name=requested_model_name)
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
