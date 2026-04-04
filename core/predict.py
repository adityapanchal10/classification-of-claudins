import numpy as np
import torch

from core.config import CLASS_MAP, DEFAULT_CLASSES


def predict_probabilities(bundle, embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.classifier.to(device)
    x = embeddings.to(device)
    with torch.no_grad():
        if bundle.uses_attention:
            logits, attn = model(x, return_attn=True)
        else:
            logits = model(x)
            attn = None
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values
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
