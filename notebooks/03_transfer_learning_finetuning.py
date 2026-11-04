"""
Multi-head fine-tuning of a Persian/multilingual encoder.

Key ideas (all justified in the Persian report):
  - TWO heads: recommendation (3-way CE) and rating_band (3-way CE).
  - MASKED losses: a row missing one target still trains the other head.
  - Different LR for encoder vs heads (encoder already knows Persian).
  - Class-weighted CE to fight imbalance.
  - Early stopping on val Macro F1 (average of both heads).
  - Save encoder+tokenizer+heads to artifacts/transformer/.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report

from config import (
    PRETRAINED_ENCODER, MAX_SEQ_LEN, BATCH_SIZE, GRAD_ACCUM,
    EPOCHS, LR_ENCODER, LR_HEAD, WEIGHT_DECAY, WARMUP_RATIO,
    SPLITS_DIR, ARTIFACTS,
    RECOMMENDATION_LABELS, RATING_BAND_LABELS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REC2ID = {l: i for i, l in enumerate(RECOMMENDATION_LABELS)}
RB2ID  = {l: i for i, l in enumerate(RATING_BAND_LABELS)}


class ReviewDataset(Dataset):
    """Row returns (input_ids, attention_mask, rec_label, band_label).
    Missing labels are returned as -100 (PyTorch's ignore_index)."""
    def __init__(self, df, tok):
        self.df = df.reset_index(drop=True)
        self.tok = tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(
            row["body"], truncation=True, max_length=MAX_SEQ_LEN,
            padding=False, return_attention_mask=True,
        )
        rec  = REC2ID.get(row["recommendation_status"], -100)
        band = RB2ID.get(row["rating_band"], -100)
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "rec_label":      rec,
            "band_label":     band,
        }


class Collator:
    def __init__(self, tok):
        self.tok = tok
    def __call__(self, batch):
        feats = self.tok.pad(
            [{"input_ids": b["input_ids"],
              "attention_mask": b["attention_mask"]} for b in batch],
            padding=True, return_tensors="pt",
        )
        feats["rec_label"]  = torch.tensor([b["rec_label"]  for b in batch])
        feats["band_label"] = torch.tensor([b["band_label"] for b in batch])
        return feats


class MultiHeadClassifier(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.rec_head  = nn.Linear(h, len(RECOMMENDATION_LABELS))
        self.band_head = nn.Linear(h, len(RATING_BAND_LABELS))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        # CLS pooling
        pooled = self.dropout(out[:, 0])
        return self.rec_head(pooled), self.band_head(pooled)


def class_weights(df, col, id_map):
    counts = df[col].dropna().value_counts()
    total  = counts.sum()
    n_cls  = len(id_map)
    w = torch.ones(n_cls)
    for lab, c in counts.items():
        w[id_map[lab]] = total / (n_cls * c)
    return w


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    rec_preds, rec_gold, band_preds, band_gold = [], [], [], []
    for b in loader:
        logits_r, logits_b = model(b["input_ids"].to(dev),
                                   b["attention_mask"].to(dev))
        rec_preds.append(logits_r.argmax(-1).cpu().numpy())
        band_preds.append(logits_b.argmax(-1).cpu().numpy())
        rec_gold.append(b["rec_label"].numpy())
        band_gold.append(b["band_label"].numpy())

    rec_preds, band_preds = map(np.concatenate, (rec_preds, band_preds))
    rec_gold,  band_gold  = map(np.concatenate, (rec_gold,  band_gold))

    m = rec_gold != -100
    rec_f1 = f1_score(rec_gold[m], rec_preds[m], average="macro") if m.any() else np.nan
    m2 = band_gold != -100
    band_f1 = f1_score(band_gold[m2], band_preds[m2], average="macro") if m2.any() else np.nan
    return {"rec_f1": rec_f1, "band_f1": band_f1,
            "avg_f1": np.nanmean([rec_f1, band_f1]),
            "rec_preds": rec_preds, "rec_gold": rec_gold,
            "band_preds": band_preds, "band_gold": band_gold}


def train_one_epoch(model, loader, opt, sched, dev,
                    rec_w, band_w, grad_accum=1):
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    total, n = 0.0, 0
    opt.zero_grad()
    for step, b in enumerate(loader):
        logits_r, logits_b = model(b["input_ids"].to(dev),
                                   b["attention_mask"].to(dev))
        loss_r = ce(logits_r, b["rec_label"].to(dev))
        # If NO row in the batch has a band label, ignore the loss
        if (b["band_label"] != -100).any():
            loss_b = ce(logits_b, b["band_label"].to(dev))
        else:
            loss_b = torch.tensor(0.0, device=dev)

        # Apply class weights via a second CE call (weighted)
        w_ce_r = nn.CrossEntropyLoss(weight=rec_w.to(dev), ignore_index=-100)
        w_ce_b = nn.CrossEntropyLoss(weight=band_w.to(dev), ignore_index=-100)
        loss = (w_ce_r(logits_r, b["rec_label"].to(dev))
                + (w_ce_b(logits_b, b["band_label"].to(dev))
                   if (b["band_label"] != -100).any() else 0.0))
        loss = loss / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        total += loss.item() * grad_accum
        n += 1
    return total / max(n, 1)


def build_optimizer(model):
    enc_params = list(model.encoder.parameters())
    head_params = list(model.rec_head.parameters()) + list(model.band_head.parameters())
    return torch.optim.AdamW([
        {"params": enc_params,  "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)


def main():
    tok = AutoTokenizer.from_pretrained(PRETRAINED_ENCODER)
    tr  = pd.read_csv(SPLITS_DIR / "train_supervised.csv")
    va  = pd.read_csv(SPLITS_DIR / "validation.csv")

    ds_tr = ReviewDataset(tr, tok)
    ds_va = ReviewDataset(va, tok)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=Collator(tok), num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=Collator(tok), num_workers=2)

    model = MultiHeadClassifier(PRETRAINED_ENCODER).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rec_w  = class_weights(tr, "recommendation_status", REC2ID)
    band_w = class_weights(tr, "rating_band", RB2ID)

    opt = build_optimizer(model)
    steps_per_epoch = len(dl_tr)
    total_steps = EPOCHS * steps_per_epoch
    sched = get_linear_schedule_with_warmup(
        opt, int(WARMUP_RATIO * total_steps), total_steps
    )

    print(f"total params: {total_params:,} | trainable: {trainable:,}")
    history = []
    best_f1, best_state = -1.0, None

    for ep in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, dl_tr, opt, sched, DEVICE,
                               rec_w, band_w, GRAD_ACCUM)
        metrics = evaluate(model, dl_va, DEVICE)
        history.append({"epoch": ep, "train_loss": loss,
                        **{k: v for k, v in metrics.items()
                           if isinstance(v, (int, float))}})
        print(history[-1])
        if metrics["avg_f1"] > best_f1:
            best_f1 = metrics["avg_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    out = ARTIFACTS / "transformer_supervised"
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out / "encoder")
    tok.save_pretrained(out / "tokenizer")
    torch.save({
        "rec_head":  model.rec_head.state_dict(),
        "band_head": model.band_head.state_dict(),
    }, out / "heads.pt")
    (out / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )

    # Full validation report
    metrics = evaluate(model, dl_va, DEVICE)
    m = metrics["rec_gold"] != -100
    print(classification_report(
        metrics["rec_gold"][m], metrics["rec_preds"][m],
        labels=list(range(len(RECOMMENDATION_LABELS))),
        target_names=RECOMMENDATION_LABELS, digits=4,
    ))
    m2 = metrics["band_gold"] != -100
    print(classification_report(
        metrics["band_gold"][m2], metrics["band_preds"][m2],
        labels=list(range(len(RATING_BAND_LABELS))),
        target_names=RATING_BAND_LABELS, digits=4,
    ))


if __name__ == "__main__":
    main()