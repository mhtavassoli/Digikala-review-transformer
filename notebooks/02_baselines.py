"""
TF-IDF + Logistic Regression baseline for BOTH heads.
Fit vocabulary + class weights on TRAIN ONLY.
This gives us:
  - a cheap reference point,
  - a signal for the error-analysis section,
  - evidence of whether a Transformer is actually needed.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score,
)

from config import SPLITS_DIR, ARTIFACTS

TRAIN = pd.read_csv(SPLITS_DIR / "train_supervised.csv")
VAL   = pd.read_csv(SPLITS_DIR / "validation.csv")
TEST  = pd.read_csv(SPLITS_DIR / "test.csv")


def fit_tfidf_lr(train, val, target_col, labels):
    mask_tr = train[target_col].notna()
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=200_000,
        sublinear_tf=True,
    )
    X_tr = vec.fit_transform(train.loc[mask_tr, "body"])

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",   # Macro F1 focus
        C=2.0,
        n_jobs=-1,
    )
    clf.fit(X_tr, train.loc[mask_tr, target_col])

    # Eval on val
    mask_va = val[target_col].notna()
    X_va = vec.transform(val.loc[mask_va, "body"])
    y_va = val.loc[mask_va, target_col]
    pred = clf.predict(X_va)
    print(f"\n=== {target_col} | VALIDATION ===")
    print(classification_report(y_va, pred, labels=labels, digits=4))
    print("Balanced accuracy:", balanced_accuracy_score(y_va, pred))
    print("Macro F1:", f1_score(y_va, pred, average="macro"))

    joblib.dump({"vec": vec, "clf": clf},
                ARTIFACTS / f"tfidf_lr_{target_col}.joblib")
    return vec, clf


for tgt, labs in [
    ("recommendation_status", ["recommended", "not_recommended", "no_idea"]),
    ("rating_band",           ["low", "medium", "high"]),
]:
    fit_tfidf_lr(TRAIN, VAL, tgt, labs)