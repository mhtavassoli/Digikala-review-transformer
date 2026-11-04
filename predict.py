"""
Offline prediction script (required by "Final Mini-Project").
Loads TF-IDF and Transformer artifacts, predicts BOTH heads,
reports probabilities, token count, and latency.
"""
import argparse
import time
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

from config import (
    ARTIFACTS, MAX_SEQ_LEN, PRETRAINED_ENCODER,
    RECOMMENDATION_LABELS, RATING_BAND_LABELS,
)
from utils.normalization import normalize_fa


class MultiHead(nn.Module):
    def __init__(self, enc, h):
        super().__init__()
        self.encoder = enc
        self.rec  = nn.Linear(h, len(RECOMMENDATION_LABELS))
        self.band = nn.Linear(h, len(RATING_BAND_LABELS))
    def forward(self, ids, am):
        x = self.encoder(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
        return self.rec(x), self.band(x)


def load_transformer():
    base = ARTIFACTS / "transformer_supervised"
    tok = AutoTokenizer.from_pretrained(base / "tokenizer")
    enc = AutoModel.from_pretrained(base / "encoder")
    model = MultiHead(enc, enc.config.hidden_size).eval()
    heads = torch.load(base / "heads.pt", map_location="cpu")
    model.rec.load_state_dict(heads["rec_head"])
    model.band.load_state_dict(heads["band_head"])
    return tok, model


def load_tfidf():
    rec  = joblib.load(ARTIFACTS / "tfidf_lr_recommendation_status.joblib")
    band = joblib.load(ARTIFACTS / "tfidf_lr_rating_band.joblib")
    return rec, band


@torch.no_grad()
def predict_transformer(text, tok, model):
    t0 = time.time()
    norm = normalize_fa(text)
    enc = tok(norm, return_tensors="pt", truncation=True,
              max_length=MAX_SEQ_LEN)
    lr, lb = model(enc["input_ids"], enc["attention_mask"])
    pr = torch.softmax(lr, -1).numpy()[0]
    pb = torch.softmax(lb, -1).numpy()[0]
    return {
        "normalized": norm,
        "tokens": int(enc["input_ids"].shape[1]),
        "recommendation": {
            "label": RECOMMENDATION_LABELS[pr.argmax()],
            "probs": {k: float(p) for k, p in zip(RECOMMENDATION_LABELS, pr)},
        },
        "rating_band": {
            "label": RATING_BAND_LABELS[pb.argmax()],
            "probs": {k: float(p) for k, p in zip(RATING_BAND_LABELS, pb)},
        },
        "latency_ms": (time.time() - t0) * 1000,
    }


def predict_tfidf(text, rec_bundle, band_bundle):
    norm = normalize_fa(text)
    X = rec_bundle["vec"].transform([norm])
    pr = rec_bundle["clf"].predict_proba(X)[0]
    Xb = band_bundle["vec"].transform([norm])
    pb = band_bundle["clf"].predict_proba(Xb)[0]
    return {
        "normalized": norm,
        "recommendation": {
            "label": RECOMMENDATION_LABELS[pr.argmax()],
            "probs": dict(zip(rec_bundle["clf"].classes_, pr.tolist())),
        },
        "rating_band": {
            "label": RATING_BAND_LABELS[pb.argmax()],
            "probs": dict(zip(band_bundle["clf"].classes_, pb.tolist())),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    tok, model = load_transformer()
    rec, band = load_tfidf()

    print("=== TF-IDF ===")
    print(predict_tfidf(args.text, rec, band))
    print("=== Transformer ===")
    print(predict_transformer(args.text, tok, model))


if __name__ == "__main__":
    main()