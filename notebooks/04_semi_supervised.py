"""
Semi-supervised experiment (recommendation head ONLY).

Procedure (per Section "Semi-Supervised Learning"):
  1. Supervised model is already trained (artifacts/transformer_supervised).
  2. Pseudo-label train_pseudo_unlabeled with P >= 0.90.
  3. Continue training the SAME architecture on
        supervised_train ∪ accepted_pseudo
     with:
        - rec_loss on both,
        - band_loss ONLY on supervised rows (masked on pseudo).
  4. Compare on test.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (
    PRETRAINED_ENCODER, MAX_SEQ_LEN, BATCH_SIZE, GRAD_ACCUM,
    PSEUDO_CONF_THRESH, PSEUDO_EPOCHS, LR_ENCODER, LR_HEAD,
    WEIGHT_DECAY, SPLITS_DIR, ARTIFACTS, RECOMMENDATION_LABELS, RATING_BAND_LABELS,
)
from notebooks.transfer_learning_finetuning import (  # reuse
    ReviewDataset, Collator, MultiHeadClassifier, build_optimizer,
    evaluate, class_weights, REC2ID, RB2ID, DEVICE,
)
from transformers import AutoTokenizer


@torch.no_grad()
def predict_proba(model, loader, dev):
    model.eval()
    probs = []
    for b in loader:
        logits_r, _ = model(b["input_ids"].to(dev),
                            b["attention_mask"].to(dev))
        probs.append(torch.softmax(logits_r, dim=-1).cpu().numpy())
    return np.concatenate(probs)


class PseudoAugDataset(Dataset):
    """
    Mixes supervised rows (with band labels) and accepted pseudo rows
    (band label = -100). Recommendation label = argmax pseudo label.
    """
    def __init__(self, sup_df, pseudo_df, tok):
        self.rows = []
        for _, r in sup_df.iterrows():
            self.rows.append({
                "body": r["body"],
                "rec":  REC2ID[r["recommendation_status"]],
                "band": RB2ID.get(r["rating_band"], -100),
            })
        for _, r in pseudo_df.iterrows():
            self.rows.append({
                "body": r["body"],
                "rec":  REC2ID[r["pseudo_label"]],
                "band": -100,
            })
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["body"], truncation=True,
                       max_length=MAX_SEQ_LEN, padding=False)
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "rec_label": r["rec"],
                "band_label": r["band"]}


def main():
    tok = AutoTokenizer.from_pretrained(PRETRAINED_ENCODER)
    sup = pd.read_csv(SPLITS_DIR / "train_supervised.csv")
    pseudo_unl = pd.read_csv(SPLITS_DIR / "train_pseudo_unlabeled.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")

    # ---- 1. Load supervised model -----------------------------------
    model = MultiHeadClassifier(PRETRAINED_ENCODER).to(DEVICE)
    sd = torch.load(ARTIFACTS / "transformer_supervised" / "heads.pt",
                    map_location=DEVICE)
    model.rec_head.load_state_dict(sd["rec_head"])
    model.band_head.load_state_dict(sd["band_head"])

    # ---- 2. Pseudo-label the unlabeled pool --------------------------
    unl_ds = ReviewDataset(pseudo_unl.assign(
        recommendation_status=None, rating_band=None), tok)
    unl_dl = DataLoader(unl_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=Collator(tok))
    probs = predict_proba(model, unl_dl, DEVICE)
    conf  = probs.max(axis=1)
    pred  = probs.argmax(axis=1)

    keep = conf >= PSEUDO_CONF_THRESH
    pseudo_unl = pseudo_unl.assign(
        pseudo_label=[RECOMMENDATION_LABELS[i] for i in pred],
        confidence=conf,
    )
    accepted = pseudo_unl[keep].copy()

    # ---- 3. Continue training ----------------------------------------
    aug_ds = PseudoAugDataset(sup, accepted, tok)
    aug_dl = DataLoader(aug_ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=Collator(tok))
    rec_w  = class_weights(sup, "recommendation_status", REC2ID)
    band_w = class_weights(sup, "rating_band", RB2ID)
    opt = build_optimizer(model)
    ce_rec  = nn.CrossEntropyLoss(weight=rec_w.to(DEVICE), ignore_index=-100)
    ce_band = nn.CrossEntropyLoss(weight=band_w.to(DEVICE), ignore_index=-100)

    for ep in range(PSEUDO_EPOCHS):
        model.train()
        opt.zero_grad()
        for step, b in enumerate(aug_dl):
            logits_r, logits_b = model(b["input_ids"].to(DEVICE),
                                       b["attention_mask"].to(DEVICE))
            loss = ce_rec(logits_r, b["rec_label"].to(DEVICE))
            if (b["band_label"] != -100).any():
                loss = loss + ce_band(logits_b, b["band_label"].to(DEVICE))
            (loss / GRAD_ACCUM).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()

    # ---- 4. Compare on test ------------------------------------------
    test_ds = ReviewDataset(test, tok)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, collate_fn=Collator(tok))

    # Supervised-only reference
    sup_model = MultiHeadClassifier(PRETRAINED_ENCODER).to(DEVICE)
    sup_model.rec_head.load_state_dict(sd["rec_head"])
    sup_model.band_head.load_state_dict(sd["band_head"])
    m_sup = evaluate(sup_model, test_dl, DEVICE)
    m_semi = evaluate(model,     test_dl, DEVICE)

    # ---- 5. Audit pseudo-label quality (ONLY for reporting) ----------
    # Ground-truth labels of masked rows are available offline in the
    # original CSV; we join them back here only to report pseudo-label
    # accuracy. They are NEVER fed to the model.
    ref = pd.read_csv(SPLITS_DIR / "train_pseudo_unlabeled.csv")
    # (Reference file for audit should be the un-masked training CSV;
    #  here we assume it exists as `train_full.csv`.)
    full = pd.read_csv(SPLITS_DIR / "train_full.csv")
    audit = accepted.merge(full[["id", "recommendation_status"]],
                           on="id", how="left")
    pl_acc = (audit["pseudo_label"] == audit["recommendation_status"]).mean()

    report = {
        "threshold": PSEUDO_CONF_THRESH,
        "unlabeled_pool_size": int(len(pseudo_unl)),
        "accepted": int(len(accepted)),
        "accepted_pct": float(len(accepted) / len(pseudo_unl)),
        "pseudo_label_class_distribution":
            accepted["pseudo_label"].value_counts().to_dict(),
        "pseudo_label_accuracy_vs_reference": float(pl_acc),
        "supervised_test": {
            "rec_f1": float(m_sup["rec_f1"]),
            "band_f1": float(m_sup["band_f1"]),
        },
        "semi_supervised_test": {
            "rec_f1": float(m_semi["rec_f1"]),
            "band_f1": float(m_semi["band_f1"]),
        },
        "delta_rec_f1": float(m_semi["rec_f1"] - m_sup["rec_f1"]),
    }
    out = ARTIFACTS / "transformer_semi_supervised"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()