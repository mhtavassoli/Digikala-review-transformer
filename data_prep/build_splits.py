"""
End-to-end data-prep driver:
  1. Load raw CSV (DuckDB for memory safety).
  2. Schema + quality gate.
  3. Parse labels, apply canonical maps.
  4. Product-grouped stratified split.
  5. Deterministic masking of training rows.
  6. Write the 6 required artifacts + split_manifest.json.
"""
import json
import duckdb
import pandas as pd
from pathlib import Path

from config import (
    SPLIT_SEED, MASK_SEED, MASK_RATE,
    RATING_BANDS, RECOMMENDATION_CANONICAL,
    SPLITS_DIR,
)
from utils.normalization import normalize_fa, is_effectively_empty
from utils.splitting import product_grouped_stratified_split
from utils.masking import mask_training_rows


def rating_to_band(rate: float) -> str | None:
    if rate is None or pd.isna(rate):
        return None
    try:
        r = float(rate)
    except (TypeError, ValueError):
        return None
    if r < 0 or r > 5:
        return None
    for lo, hi, label, _ in RATING_BANDS:
        if lo <= r < hi or (label == "high" and r == hi):
            return label
    return None


def load_and_clean(csv_path: str) -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect()
    # DuckDB reads CSV with real quoting; avoids the "row count != line count" trap
    raw = con.execute(f"SELECT * FROM read_csv_auto('{csv_path}', header=True)").df()

    audit = {"raw_rows": len(raw), "columns": list(raw.columns)}

    # --- Quality gate -------------------------------------------------
    # 1. Drop rows with null / empty body (they can't be predicted)
    raw["body"] = raw["body"].fillna("").astype(str).map(normalize_fa)
    raw = raw[~raw["body"].map(is_effectively_empty)]

    # 2. Parse rate
    raw["rate_num"] = pd.to_numeric(raw["rate"], errors="coerce")
    invalid_rate = raw["rate_num"].notna() & ((raw["rate_num"] < 0) | (raw["rate_num"] > 5))
    audit["invalid_rate_count"] = int(invalid_rate.sum())
    raw.loc[invalid_rate, "rate_num"] = None
    raw.loc[raw["rate_num"].isna(), "rate_num"] = None

    # 3. Canonical recommendation label
    raw["recommendation_status"] = (
        raw["recommendation_status"]
        .astype(str).str.strip()
        .map(RECOMMENDATION_CANONICAL)   # extend map after auditing real values
    )

    # 4. Derive rating_band
    raw["rating_band"] = raw["rate_num"].map(rating_to_band)

    # 5. Drop rows with BOTH targets missing (they can only sit in natural_unlabeled,
    #    which is optional; we drop here and re-add in an optional step)
    raw = raw[~(
        raw["recommendation_status"].isna() & raw["rating_band"].isna()
    )].copy()

    return raw, audit


def main(csv_path: str):
    df, audit = load_and_clean(csv_path)

    # --- Split --------------------------------------------------------
    df = product_grouped_stratified_split(
        df,
        product_col="product_id",
        strat_col="recommendation_status",
        ratios=(0.70, 0.15, 0.15),
        seed=SPLIT_SEED,
    )

    train = df[df["split"] == "train"].copy()
    val   = df[df["split"] == "val"].copy()
    test  = df[df["split"] == "test"].copy()

    # --- Masking ------------------------------------------------------
    train_sup, train_pseudo_unl = mask_training_rows(
        train,
        id_col="id",
        label_col="recommendation_status",
        mask_rate=MASK_RATE,
    )

    # --- Persist ------------------------------------------------------
    keep_cols = ["id", "body", "product_id", "recommendation_status", "rating_band", "rate_num"]
    train_sup[keep_cols].to_csv(SPLITS_DIR / "train_supervised.csv", index=False)
    train_pseudo_unl.to_csv(SPLITS_DIR / "train_pseudo_unlabeled.csv", index=False)
    val[keep_cols].to_csv(SPLITS_DIR / "validation.csv", index=False)
    test[keep_cols].to_csv(SPLITS_DIR / "test.csv", index=False)

    # natural_unlabeled.csv — optional (non-masked rows that never had a label)
    # (skipped here; can be added if you want the optional extension)

    manifest = {
        "split_seed": SPLIT_SEED,
        "mask_seed": MASK_SEED,
        "mask_rate": MASK_RATE,
        "rating_band_thresholds": [
            {"low": lo, "high": hi, "label": lb, "ordinal": od}
            for lo, hi, lb, od in RATING_BANDS
        ],
        "recommendation_labels": list(RECOMMENDATION_CANONICAL.values()),
        "split_strategy": "product_grouped_stratified",
        "audit": audit,
        "counts": {
            "train_supervised": int(len(train_sup)),
            "train_pseudo_unlabeled": int(len(train_pseudo_unl)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "masked_per_class": (
            train_pseudo_unl.merge(
                train[["id", "recommendation_status"]], on="id", how="left"
            )["recommendation_status"].value_counts().to_dict()
        ),
    }
    (SPLITS_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])