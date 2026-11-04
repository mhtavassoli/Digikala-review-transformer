"""
Deterministic pseudo-label masking, exactly per Section "Public and
Reproducible Masking Protocol".
"""
import hashlib
import pandas as pd

MASK_SEED = "digikala-semi-supervised-v1"
MASK_RATE = 0.20


def stable_hash(row_id: str) -> int:
    key = f"{MASK_SEED}|{row_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def mask_training_rows(
    train_df: pd.DataFrame,
    id_col: str = "id",
    label_col: str = "recommendation_status",
    mask_rate: float = MASK_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within each recommendation class, sort by stable_hash(id) and mask
    exactly floor(mask_rate * class_count) rows.

    Returns (supervised_train, pseudo_unlabeled).
    Pseudo-unlabeled keeps ONLY id and body (labels removed).
    """
    supervised_idx, masked_idx = [], []
    for label, group in train_df.groupby(label_col, dropna=True):
        n = len(group)
        k = int(mask_rate * n)   # floor
        order = group.assign(_h=group[id_col].apply(stable_hash)) \
                     .sort_values("_h")
        masked = order.head(k).index
        kept   = order.tail(n - k).index
        masked_idx.extend(masked.tolist())
        supervised_idx.extend(kept.tolist())

    sup = train_df.loc[supervised_idx].copy()
    pse = train_df.loc[masked_idx][[id_col, "body"]].copy()
    return sup, pse