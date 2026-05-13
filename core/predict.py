import numpy as np
import torch

from core.config import CLASS_MAP, DEFAULT_CLASSES


def resolve_residue_slice(cfg):
    residue_slice = cfg.get("residue_slice") if isinstance(cfg, dict) else None
    if not residue_slice:
        return None
    if isinstance(residue_slice, dict):
        start = residue_slice.get("start", 0)
        end = residue_slice.get("end")
        start = int(start)
        end = int(end) if end is not None else None
        return (start, end)
    if isinstance(residue_slice, (list, tuple)) and residue_slice and isinstance(residue_slice[0], (list, tuple)):
        resolved = []
        for start, end in residue_slice:
            start = int(start)
            end = int(end) if end is not None else None
            resolved.append((start, end))
        return resolved
    start, end = residue_slice
    start = int(start)
    end = int(end) if end is not None else None
    return (start, end)


def slice_embeddings(embeddings, residue_slice):
    if residue_slice is None or embeddings is None:
        return embeddings
    if not hasattr(embeddings, "shape") or len(embeddings.shape) < 2:
        return embeddings
    if isinstance(residue_slice, (list, tuple)) and residue_slice and isinstance(residue_slice[0], (list, tuple)):
        chunks = []
        seq_len = embeddings.shape[1]
        for start, end in residue_slice:
            start = max(0, min(int(start), seq_len))
            if end is None:
                end = seq_len
            else:
                end = max(start, min(int(end), seq_len))
            chunks.append(embeddings[:, start:end, :])
        return torch.cat(chunks, dim=1) if chunks else embeddings[:, :0, :]
    start, end = residue_slice
    seq_len = embeddings.shape[1]
    start = max(0, min(int(start), seq_len))
    if end is None:
        end = seq_len
    else:
        end = max(start, min(int(end), seq_len))
    return embeddings[:, start:end, :]


def slice_sequence(sequence, residue_slice):
    if residue_slice is None or sequence is None:
        return sequence
    if isinstance(residue_slice, (list, tuple)) and residue_slice and isinstance(residue_slice[0], (list, tuple)):
        seq_len = len(sequence)
        chunks = []
        for start, end in residue_slice:
            start = max(0, min(int(start), seq_len))
            if end is None:
                end = seq_len
            else:
                end = max(start, min(int(end), seq_len))
            chunks.append(sequence[start:end])
        return "".join(chunks)
    start, end = residue_slice
    seq_len = len(sequence)
    start = max(0, min(int(start), seq_len))
    if end is None:
        end = seq_len
    else:
        end = max(start, min(int(end), seq_len))
    return sequence[start:end]


def predict_probabilities(bundle, embeddings, return_attention=True):
    print(f"[PRED] Start model={bundle.model_name} n_seq={int(embeddings.shape[0])} dtype={embeddings.dtype}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.classifier.to(device)
    x = embeddings.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if bundle.uses_attention and return_attention:
            logits, attn = model(x, return_attn=True)
        else:
            logits = model(x)
            attn = None
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values
    pred_counts = {CLASS_MAP[i]: int((preds == i).sum()) for i in CLASS_MAP}
    print(f"[PRED] Done model={bundle.model_name} preds={pred_counts}")
    return preds.cpu().numpy(), confs.cpu().numpy(), probs.cpu().numpy(), attn.cpu() if attn is not None else None


def build_prediction_table(df_valid, preds, confs, probs):
    out = df_valid.copy().reset_index(drop=True)
    # remove columns: description, sequence, length, is_valid, invalid_chars
    out = out.drop(columns=["description", "sequence", "length", "is_valid", "invalid_chars"], errors="ignore")
    out["predicted_class"] = [CLASS_MAP[int(i)] for i in preds]
    out["confidence"] = confs
    for idx, cls_name in enumerate(DEFAULT_CLASSES):
        out[f"prob_{cls_name}"] = probs[:, idx]
    return out
