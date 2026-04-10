import numpy as np
import torch

from core.config import CLASS_MAP, DEFAULT_CLASSES


def predict_probabilities(bundle, embeddings, return_attention=True):
    print(f"[PRED] Start model={bundle.model_name} n_seq={int(embeddings.shape[0])}")
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
    print(f"[PRED] Done model={bundle.model_name}")
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
