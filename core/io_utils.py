import io
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from Bio import SeqIO


@dataclass
class SequenceDataset:
    records: List[Tuple[str, str]]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for idx, (desc, seq) in enumerate(self.records):
            clean = str(seq).strip().upper().replace(" ", "")
            rows.append({
                "seq_id": desc or f"sequence_{idx + 1}",
                "description": desc,
                "sequence": clean,
                "length": len(clean),
            })
        return pd.DataFrame(rows)


def parse_fasta_text(content: str) -> pd.DataFrame:
    handle = io.StringIO(content)
    records = [(rec.description, str(rec.seq)) for rec in SeqIO.parse(handle, "fasta")]
    return SequenceDataset(records).to_frame()


def parse_plain_text_sequences(content: str) -> pd.DataFrame:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    records = [(f"sequence_{i + 1}", line) for i, line in enumerate(lines)]
    return SequenceDataset(records).to_frame()


def detect_input_dataframe(text_value: str, uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        print(f"[IO] Input source=file name={uploaded_file.name}")
        content = uploaded_file.read().decode("utf-8")
        df = parse_fasta_text(content)
        print(f"[IO] Parsed records={len(df)}")
        return df
    if text_value.strip().startswith(">"):
        print("[IO] Input source=text_fasta")
        df = parse_fasta_text(text_value)
        print(f"[IO] Parsed records={len(df)}")
        return df
    print("[IO] Input source=text_lines")
    df = parse_plain_text_sequences(text_value)
    print(f"[IO] Parsed records={len(df)}")
    return df


def validate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO-*")
    out = df.copy()
    out["sequence"] = out["sequence"].astype(str).str.upper()
    out["is_valid"] = out["sequence"].apply(lambda s: len(s) > 0 and set(s).issubset(allowed))
    out["invalid_chars"] = out["sequence"].apply(lambda s: "".join(sorted(set([c for c in s if c not in allowed]))))
    print(f"[IO] Valid sequences={int(out['is_valid'].sum())}/{len(out)}")
    return out
