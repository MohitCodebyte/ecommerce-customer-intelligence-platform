"""
Production-level XGBoost training pipeline for the
E-Commerce Customer Intelligence & Repeat Purchase Propensity System.

Key design decisions (read this before you run it):

1. This dataset has EXTREME class imbalance (~1.1-1.5% positive rate) AND
   the raw ROC-AUC in earlier experiments was only ~0.55-0.60. That means the
   available RFM-style features carry weak signal. No amount of SMOTE /
   scale_pos_weight / threshold tuning can create signal that isn't there.
   This script:
     - adds missing-value indicator flags (the purchase-gap columns are
       ~97% missing and that missingness itself is informative)
     - adds ratio / rate engineered features
     - tunes hyperparameters against PR-AUC (average_precision), which is
       the right ranking metric under heavy imbalance (F1/F2 on a fixed
       0.5 threshold is meaningless here)
     - selects the decision threshold ONLY on a validation split carved out
       of training data, never on the untouched test set
     - lets you set a target recall and finds the threshold that maximizes
       precision subject to that recall (a realistic production knob)
     - saves model + imputer + feature list + metrics + threshold as
       versioned artifacts, with logging, so it's runnable as a CLI job.

2. If after running this your precision is still very low at a usable
   recall, the fix is NOT more hyperparameter search. It's better features:
   product category, seller/geolocation info, payment method/installments,
   review scores, acquisition channel, day-of-week/seasonality, and
   customer-level embeddings from order sequences. Feature signal is the
   ceiling; the model just gets you close to that ceiling.

Usage:
    python train_repeat_purchase_xgboost.py \
        --train-file data/processed/repeat_purchase_train.csv \
        --test-file  data/processed/repeat_purchase_test.csv \
        --target-recall 0.40 \
        --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("repeat_purchase_pipeline")

TARGET_COL = "repeat_purchase"

DROP_COLUMNS = [
    "customer_unique_id",
    "snapshot_date",
    "future_end_date",
    "first_purchase_date",
    "last_purchase_date",
    "future_purchase_count",
    "churn_label",
    TARGET_COL,
]

# Columns whose missingness is itself informative (e.g. purchase-gap stats
# only exist for customers who already had >1 historical order).
GAP_COLUMNS = [
    "mean_purchase_gap_days",
    "median_purchase_gap_days",
    "max_purchase_gap_days",
    "purchase_gap_count",
]


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missing indicator flags BEFORE imputation destroys the signal
    for col in GAP_COLUMNS:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)

    safe_orders = df["total_orders"].replace(0, np.nan)
    safe_lifetime = df["customer_lifetime_days"].replace(0, np.nan)
    safe_window = df["observation_window_days"].replace(0, np.nan)

    df["revenue_per_order"] = df["total_revenue"] / safe_orders
    df["orders_per_lifetime_day"] = df["total_orders"] / safe_lifetime
    df["revenue_per_lifetime_day"] = df["total_revenue"] / safe_lifetime
    df["recency_ratio"] = df["recency_days"] / safe_window
    df["purchase_frequency_ratio"] = df["purchase_frequency"] / safe_window
    df["gap_ratio"] = df["mean_purchase_gap_days"] / safe_window
    df["revenue_order_frequency"] = df["total_revenue"] / (df["total_orders"] + 1)

    # Log transforms tame heavy right-skew in revenue/order counts
    for col in ["total_revenue", "average_order_value", "total_orders", "recency_days"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # Extra interaction features -- squeeze what little signal exists
    # a bit harder out of the base RFM columns.
    if {"recency_days", "customer_lifetime_days"} <= set(df.columns):
        df["recency_to_lifetime"] = df["recency_days"] / safe_lifetime
    if {"total_orders", "purchase_frequency"} <= set(df.columns):
        df["orders_x_frequency"] = df["total_orders"] * df["purchase_frequency"]
    if {"repeat_customer", "recency_days"} <= set(df.columns):
        df["repeat_customer_recency"] = df["repeat_customer"] * df["recency_days"]
    if {"average_order_value", "recency_days"} <= set(df.columns):
        df["aov_recency_interaction"] = df["average_order_value"] / (df["recency_days"] + 1)

    return df


def build_feature_matrix(df: pd.DataFrame, feature_columns: Optional[list[str]] = None):
    df = engineer_features(df)
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    if feature_columns is not None:
        # align test/inference columns to training columns exactly
        for col in feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feature_columns]
    return X


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    train_file: str
    test_file: str
    output_dir: str = "artifacts"
    optimization_mode: str = "recall_floor"  # "recall_floor" or "precision_floor"
    target_recall: float = 0.40
    target_precision: float = 0.05
    n_iter: int = 40
    cv_folds: int = 5
    random_state: int = 42
    validation_size: float = 0.20
    calibrate: bool = True
    n_ensemble_seeds: int = 3


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(config: TrainConfig) -> dict:
    out_dir = Path(config.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    log.info("Loading data ...")
    train_df = pd.read_csv(config.train_file)
    test_df = pd.read_csv(config.test_file)

    y_train_full = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    X_train_full = build_feature_matrix(train_df)
    feature_columns = list(X_train_full.columns)
    X_test = build_feature_matrix(test_df, feature_columns=feature_columns)

    log.info(f"Features used ({len(feature_columns)}): {feature_columns}")
    log.info(
        f"Train positive rate: {y_train_full.mean():.4%} | "
        f"Test positive rate: {y_test.mean():.4%}"
    )

    # Carve a validation split out of training data ONLY, for threshold
    # selection. Test set stays untouched until the very final evaluation.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.validation_size,
        stratify=y_train_full,
        random_state=config.random_state,
    )

    imputer = SimpleImputer(strategy="median")
    X_fit_imp = pd.DataFrame(
        imputer.fit_transform(X_fit), columns=feature_columns, index=X_fit.index
    )
    X_valid_imp = pd.DataFrame(
        imputer.transform(X_valid), columns=feature_columns, index=X_valid.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=feature_columns, index=X_test.index
    )

    # ---- Hyperparameter search on the FIT split (not full training data),
    # scored on PR-AUC (average_precision), which is the correct ranking
    # metric for severe imbalance -- unlike F1/F2 on a fixed threshold.
    pos_rate = y_fit.mean()
    base_spw = max(1.0, (1 - pos_rate) / pos_rate)  # theoretical balance point

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.05, 0.1, 0.25, 0.5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1],
        "reg_lambda": [1, 3, 5, 10, 20],
        "scale_pos_weight": sorted(
            {round(base_spw * m, 1) for m in (0.5, 0.75, 1.0, 1.5, 2.0)}
        ),
    }

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=config.n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=config.random_state,
        refit=True,
    )

    log.info("Starting hyperparameter search (scoring = PR-AUC) ...")
    search.fit(X_fit_imp, y_fit)
    log.info(f"Best CV PR-AUC: {search.best_score_:.4f}")
    log.info(f"Best params: {search.best_params_}")

    best_params = search.best_params_

    # ---- Small multi-seed ensemble: average probabilities from several
    # XGBoost models trained with the tuned params but different random
    # seeds. This does NOT fix a weak-signal ceiling, but it does reduce
    # prediction variance, which in turn reduces the number of confident-
    # but-wrong false positives near the decision boundary -- a real,
    # if modest, precision gain.
    def ensemble_predict_proba(fit_X, fit_y, predict_X, seeds):
        probs = np.zeros(len(predict_X))
        for seed in seeds:
            params = dict(best_params)
            m = XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                n_jobs=-1,
                random_state=seed,
                **params,
            )
            if config.calibrate:
                # isotonic calibration makes predicted probabilities map
                # more faithfully to true positive rate, which makes the
                # precision/recall trade-off at a given threshold more
                # reliable (not just "higher" -- more *trustworthy*).
                m = CalibratedClassifierCV(m, method="isotonic", cv=3)
            m.fit(fit_X, fit_y)
            probs += m.predict_proba(predict_X)[:, 1]
        return probs / len(seeds)

    seeds = [config.random_state + i for i in range(config.n_ensemble_seeds)]

    log.info(f"Training {len(seeds)}-seed ensemble for validation threshold selection ...")
    valid_prob = ensemble_predict_proba(X_fit_imp, y_fit, X_valid_imp, seeds)

    # ---- Threshold selection on the validation split only
    precisions, recalls, thresholds = precision_recall_curve(y_valid, valid_prob)
    # precision_recall_curve returns one more point than thresholds; align them
    precisions, recalls = precisions[:-1], recalls[:-1]

    if config.optimization_mode == "precision_floor":
        # Business case: don't flag a customer unless the model is right
        # often enough to act on it. Maximize recall subject to a minimum
        # acceptable precision.
        candidates = [
            (t, p, r)
            for t, p, r in zip(thresholds, precisions, recalls)
            if p >= config.target_precision
        ]
        if candidates:
            best_threshold, best_precision, best_recall = max(candidates, key=lambda x: x[2])
        else:
            log.warning(
                f"No threshold on validation reached target precision "
                f"{config.target_precision:.2%}. Using the highest-precision "
                f"threshold available instead -- this is the feature ceiling, "
                f"not a bug."
            )
            best_idx = int(np.argmax(precisions))
            best_threshold = thresholds[best_idx]
            best_precision, best_recall = precisions[best_idx], recalls[best_idx]
    else:
        # recall_floor: maximize precision subject to a minimum recall
        candidates = [
            (t, p, r)
            for t, p, r in zip(thresholds, precisions, recalls)
            if r >= config.target_recall
        ]
        if candidates:
            best_threshold, best_precision, best_recall = max(candidates, key=lambda x: x[1])
        else:
            log.warning(
                f"No threshold on validation reached target recall "
                f"{config.target_recall:.2f}. Falling back to best-F2 threshold."
            )
            f2_scores = [
                fbeta_score(y_valid, (valid_prob >= t).astype(int), beta=2, zero_division=0)
                for t in thresholds
            ]
            best_idx = int(np.argmax(f2_scores))
            best_threshold = thresholds[best_idx]
            best_precision, best_recall = precisions[best_idx], recalls[best_idx]

    log.info(
        f"Selected threshold={best_threshold:.4f} "
        f"(validation precision={best_precision:.4f}, recall={best_recall:.4f})"
    )

    # ---- Refit the same ensemble on ALL training data (fit + valid),
    # then evaluate once on the untouched test set.
    X_train_full_imp = pd.DataFrame(
        imputer.fit_transform(X_train_full), columns=feature_columns, index=X_train_full.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=feature_columns, index=X_test.index
    )
    log.info("Refitting ensemble on full training data for final test evaluation ...")
    test_prob = ensemble_predict_proba(X_train_full_imp, y_train_full, X_test_imp, seeds)
    test_pred = (test_prob >= best_threshold).astype(int)

    metrics = {
        "threshold": float(best_threshold),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "f2": float(fbeta_score(y_test, test_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "test_prevalence": float(y_test.mean()),
        "cv_best_pr_auc": float(search.best_score_),
        "best_params": search.best_params_,
    }

    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    log.info("=" * 70)
    log.info("FINAL TEST RESULTS")
    log.info("=" * 70)
    for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]:
        log.info(f"{k:>18}: {metrics[k]:.4f}")
    log.info(f"PR-AUC vs baseline prevalence: {metrics['pr_auc']:.4f} vs {metrics['test_prevalence']:.4f}")
    if metrics["roc_auc"] < 0.65:
        log.warning(
            "ROC-AUC is still low (<0.65). This means current features carry "
            "weak signal for this target -- consider adding product/category, "
            "payment, geography, or review-score features before further tuning."
        )

    # ---- Refit and save each seed model of the final ensemble so inference
    # can reload and average them exactly the same way.
    final_models = []
    for seed in seeds:
        params = dict(best_params)
        m = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            **params,
        )
        if config.calibrate:
            m = CalibratedClassifierCV(m, method="isotonic", cv=3)
        m.fit(X_train_full_imp, y_train_full)
        final_models.append(m)

    joblib.dump(final_models, out_dir / "models" / "xgb_repeat_purchase_ensemble.joblib")
    joblib.dump(imputer, out_dir / "models" / "imputer.joblib")
    with open(out_dir / "models" / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(out_dir / "reports" / "metrics.json", "w") as f:
        json.dump({"config": asdict(config), **metrics}, f, indent=2)

    log.info(f"Artifacts saved to: {out_dir.resolve()}")
    return metrics


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument(
        "--optimization-mode",
        choices=["recall_floor", "precision_floor"],
        default="recall_floor",
        help=(
            "recall_floor: maximize precision subject to --target-recall. "
            "precision_floor: maximize recall subject to --target-precision "
            "(use this when precision is the hard production requirement)."
        ),
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.40,
        help="Used in recall_floor mode: minimum recall to hit.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.05,
        help="Used in precision_floor mode: minimum precision to hit.",
    )
    parser.add_argument("--n-iter", type=int, default=40, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument(
        "--no-calibrate", action="store_true", help="Disable isotonic probability calibration"
    )
    parser.add_argument(
        "--n-ensemble-seeds", type=int, default=3, help="Number of seeds to average for the final model"
    )
    args = parser.parse_args(argv)
    return TrainConfig(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        optimization_mode=args.optimization_mode,
        target_recall=args.target_recall,
        target_precision=args.target_precision,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        validation_size=args.validation_size,
        calibrate=not args.no_calibrate,
        n_ensemble_seeds=args.n_ensemble_seeds,
    )


def main(argv=None):
    config = parse_args(argv)
    try:
        train(config)
    except Exception:
        log.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()