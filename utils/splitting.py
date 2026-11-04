"""
Product-grouped train/val/test split with class representativeness.

Why group by product?
    A single product can have hundreds of reviews. If we split by row, the
    same product (and thus very similar sentences) appears in train and test
    → optimistic bias. Grouping by product_id prevents this leakage.

Why make splits "representative"?
    A pure random group split can accidentally put all "not_recommended"
    reviews of popular products into test, breaking class ratios.
    We solve this with a stratified greedy bin-packing: for each stratum
    (dominant recommendation class), assign products to the split with the
    largest deficit to its target ratio.
"""
import numpy as np
import pandas as pd


def product_grouped_stratified_split(
    df: pd.DataFrame,
    product_col: str = "product_id",
    strat_col: str = "recommendation_status",
    ratios=(0.70, 0.15, 0.15),
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Returns df with an extra column `split` ∈ {'train','val','test'}.
    Products are guaranteed to be in exactly one split.
    """
    rng = np.random.default_rng(seed)
    split_names = ["train", "val", "test"]

    # Per-product dominant stratum label and size
    def _mode(s: pd.Series):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else "MISSING"

    prod = (
        df.groupby(product_col)
          .agg(n=(strat_col, "size"), label=(strat_col, _mode))
          .reset_index()
    )

    # Shuffle products once (reproducible)
    prod = prod.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    prod["split"] = None

    # For each stratum, do greedy stratified assignment
    for label in prod["label"].unique():
        sub = prod[prod["label"] == label].copy()
        # Sort descending: big products first for better packing
        sub = sub.sort_values("n", ascending=False)

        sub_total = int(sub["n"].sum())
        target = {sp: r * sub_total for sp, r in zip(split_names, ratios)}
        current = {sp: 0 for sp in split_names}

        for idx, row in sub.iterrows():
            # Choose the split with the LARGEST remaining deficit
            deficits = {sp: target[sp] - current[sp] for sp in split_names}
            chosen = max(deficits, key=deficits.get)
            prod.at[idx, "split"] = chosen
            current[chosen] += row["n"]

    return df.merge(prod[[product_col, "split"]], on=product_col, how="left")