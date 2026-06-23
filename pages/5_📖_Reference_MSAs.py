import re
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import BASE_DIR, CLASS_MAP
from core.ui import global_sidebar

st.set_page_config(page_title="Reference MSAs", layout="wide", page_icon="🧬")
st.logo("🧬")

global_sidebar()

st.title("Reference MSAs")
st.markdown(
    "The four reference MSAs used for alignment-aware embedding. "
    "Sequences are drawn from the training set and cover all three functional classes."
)

REFERENCE_MSA_DIR = BASE_DIR / "reference_msas"

REFERENCE_MSAS = {
    "Full sequences — Balanced": REFERENCE_MSA_DIR / "ref_full_seqs_msa_balanced.fasta",
    "Full sequences — Diverse":  REFERENCE_MSA_DIR / "ref_full_seqs_msa_diverse.fasta",
    "ECS only — Balanced":       REFERENCE_MSA_DIR / "ref_ecs_only_msa_balanced.fasta",
    "ECS only — Diverse":        REFERENCE_MSA_DIR / "ref_ecs_only_msa_diverse.fasta",
}

# Reverse-map numeric class keys to readable labels
CLASS_LABELS = {
    "barrier":  CLASS_MAP[0],   # "Barrier forming"
    "cation":   CLASS_MAP[1],   # "Cation-channel forming"
    "anion":    CLASS_MAP[2],   # "Anion-channel forming"
}

# Known per-claudin-family functional class assignments
CLDN_CLASS: dict[str, str] = {
    "cldn1":   "barrier",
    "cldn2":   "cation",
    "cldn3":   "barrier",
    "cldn5":   "barrier",
    "cldn10a": "anion",
    "cldn10b": "cation",
    "cldn14":  "barrier",
    "cldn15":  "cation",
}


def _parse_fasta_headers(fasta_path: Path) -> pd.DataFrame:
    """Return a DataFrame with one row per sequence: seq_id, cldn_family, func_class."""
    rows = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">"):
                continue
            seq_id = line[1:].split()[0]
            m = re.search(r"major_label=(\S+)", line)
            family = m.group(1).lower() if m else "unknown"
            func_key = CLDN_CLASS.get(family, "unknown")
            func_label = CLASS_MAP.get(
                {"barrier": 0, "cation": 1, "anion": 2}.get(func_key, -1),
                "Unknown"
            )
            rows.append({"seq_id": seq_id, "cldn_family": family, "functional_class": func_label})
    return pd.DataFrame(rows)


for msa_label, msa_path in REFERENCE_MSAS.items():
    with st.expander(f"**{msa_label}**  —  `{msa_path.name}`", expanded=False):
        if not msa_path.exists():
            st.error(f"File not found: `{msa_path}`")
            continue

        df = _parse_fasta_headers(msa_path)

        # ── Top-level stats ──────────────────────────────────────────────
        n_total = len(df)
        n_families = df["cldn_family"].nunique()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total sequences", n_total)
        col2.metric("Claudin families", n_families)
        for col, (key, label) in zip([col3, col4, col5], CLASS_LABELS.items()):
            count = int((df["functional_class"] == CLASS_MAP[{"barrier": 0, "cation": 1, "anion": 2}[key]]).sum())
            col.metric(label, count)

        st.markdown("#### FASTA contents")
        fasta_text = msa_path.read_text()
        st.container(height=300).code(fasta_text, language=None)

        st.markdown("#### Per-family breakdown")

        # Pivot: rows = cldn_family, cols = functional class counts
        family_counts = (
            df.groupby(["cldn_family", "functional_class"])
            .size()
            .unstack(fill_value=0)
            .rename_axis(None, axis=1)
            .reset_index()
            .rename(columns={"cldn_family": "Claudin family"})
        )
        # Add total column
        class_cols = [c for c in family_counts.columns if c != "Claudin family"]
        family_counts["Total"] = family_counts[class_cols].sum(axis=1)
        family_counts = family_counts.sort_values("Claudin family").reset_index(drop=True)

        st.dataframe(family_counts, width='stretch', hide_index=True)

with st.expander("ℹ️ Balanced vs Diverse — what's the difference?", expanded=False):
    st.markdown("""
**Balanced MSA:**
Every chunk contains an exactly equal number of sequences from each claudin family
(`chunk_size ÷ n_families` per family). This gives the MSA Transformer a uniform,
unbiased view of all families in every forward pass.

**Diverse MSA:**
Every chunk is seeded with at least one sequence from every family (guaranteeing full
family coverage), and then the remaining slots are filled round-robin, prioritising
families with the most leftover sequences. This means larger families (in training)
contribute more context rows while still ensuring no family is ever absent. 

**Which to use?**
- Use **Balanced** when you want equal attention from all families.
- Use **Diverse** when you want the embedding to reflect a more training-like sequence distribution 
""")
