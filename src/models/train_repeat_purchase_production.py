"""
Production XGBoost Pipeline
E-Commerce Customer Intelligence & Repeat Purchase Propensity System

Pipeline:
1. Load train/test data
2. Feature engineering
3. Train/validation split
4. PR-AUC based hyperparameter tuning
5. Validation threshold selection
6. Final model refit on full training data
7. Untouched test evaluation
8. Save production artifacts
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

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from xgboost import XGBClassifier


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("production_repeat_purchase")


# ============================================================
# CONSTANTS
# ============================================================

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

GAP_COLUMNS = [
    "mean_purchase_gap_days",
    "median_purchase_gap_days",
    "max_purchase_gap_days",
    "purchase_gap_count",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Missing-value indicators
    # --------------------------------------------------------

    for col in GAP_COLUMNS:

        if col in df.columns:

            df[f"{col}_is_missing"] = (
                df[col].isna().astype(int)
            )

    # --------------------------------------------------------
    # Safe denominators
    # --------------------------------------------------------

    safe_orders = (
        df["total_orders"]
        .replace(0, np.nan)
    )

    safe_lifetime = (
        df["customer_lifetime_days"]
        .replace(0, np.nan)
    )

    safe_window = (
        df["observation_window_days"]
        .replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Ratio / rate features
    # --------------------------------------------------------

    df["revenue_per_order"] = (
        df["total_revenue"] /
        safe_orders
    )

    df["orders_per_lifetime_day"] = (
        df["total_orders"] /
        safe_lifetime
    )

    df["revenue_per_lifetime_day"] = (
        df["total_revenue"] /
        safe_lifetime
    )

    df["recency_ratio"] = (
        df["recency_days"] /
        safe_window
    )

    df["purchase_frequency_ratio"] = (
        df["purchase_frequency"] /
        safe_window
    )

    df["gap_ratio"] = (
        df["mean_purchase_gap_days"] /
        safe_window
    )

    df["revenue_order_frequency"] = (
        df["total_revenue"] /
        (df["total_orders"] + 1)
    )

    # --------------------------------------------------------
    # Log transformations
    # --------------------------------------------------------

    for col in [
        "total_revenue",
        "average_order_value",
        "total_orders",
        "recency_days",
    ]:

        if col in df.columns:

            df[f"log_{col}"] = np.log1p(
                df[col].clip(lower=0)
            )

    # --------------------------------------------------------
    # Interaction features
    # --------------------------------------------------------

    if {
        "recency_days",
        "customer_lifetime_days",
    }.issubset(df.columns):

        df["recency_to_lifetime"] = (
            df["recency_days"] /
            safe_lifetime
        )

    if {
        "total_orders",
        "purchase_frequency",
    }.issubset(df.columns):

        df["orders_x_frequency"] = (
            df["total_orders"] *
            df["purchase_frequency"]
        )

    if {
        "repeat_customer",
        "recency_days",
    }.issubset(df.columns):

        df["repeat_customer_recency"] = (
            df["repeat_customer"] *
            df["recency_days"]
        )

    if {
        "average_order_value",
        "recency_days",
    }.issubset(df.columns):

        df["aov_recency_interaction"] = (
            df["average_order_value"] /
            (df["recency_days"] + 1)
        )

    return df


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
):

    df = engineer_features(df)

    X = df.drop(
        columns=[
            c
            for c in DROP_COLUMNS
            if c in df.columns
        ],
        errors="ignore",
    )

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Align test/inference columns
    # exactly with training columns
    if feature_columns is not None:

        for col in feature_columns:

            if col not in X.columns:

                X[col] = np.nan

        X = X[feature_columns]

    return X


# ============================================================
# CONFIG
# ============================================================

@dataclass
class TrainConfig:

    train_file: str

    test_file: str

    output_dir: str = "artifacts_production"

    target_recall: float = 0.40

    n_iter: int = 40

    cv_folds: int = 5

    random_state: int = 42

    validation_size: float = 0.20

    n_ensemble_seeds: int = 3


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

def create_model(
    params: dict,
    seed: int,
):

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **params,
    )


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict_proba(
    X_train,
    y_train,
    X_predict,
    best_params,
    seeds,
):

    probabilities = np.zeros(
        len(X_predict),
        dtype=float,
    )

    for seed in seeds:

        model = create_model(
            best_params,
            seed,
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities += (
            model.predict_proba(
                X_predict
            )[:, 1]
        )

    probabilities /= len(seeds)

    return probabilities


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def select_threshold(
    y_valid,
    valid_prob,
    target_recall,
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_valid,
            valid_prob,
        )
    )

    # precision_recall_curve returns
    # one extra precision/recall point
    precision = precision[:-1]
    recall = recall[:-1]

    candidates = []

    for threshold, p, r in zip(
        thresholds,
        precision,
        recall,
    ):

        if r >= target_recall:

            candidates.append(
                (
                    threshold,
                    p,
                    r,
                )
            )

    if candidates:

        # Maximize precision
        # while maintaining target recall

        best_threshold, best_precision, best_recall = max(
            candidates,
            key=lambda x: x[1],
        )

    else:

        log.warning(
            f"No validation threshold "
            f"reached target recall "
            f"{target_recall:.2%}."
        )

        log.warning(
            "Falling back to best F2 threshold."
        )

        f2_scores = []

        for threshold in thresholds:

            predictions = (
                valid_prob >= threshold
            ).astype(int)

            score = fbeta_score(
                y_valid,
                predictions,
                beta=2,
                zero_division=0,
            )

            f2_scores.append(score)

        best_index = int(
            np.argmax(f2_scores)
        )

        best_threshold = thresholds[
            best_index
        ]

        best_precision = precision[
            best_index
        ]

        best_recall = recall[
            best_index
        ]

    return (
        float(best_threshold),
        float(best_precision),
        float(best_recall),
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def create_threshold_table(
    y_valid,
    valid_prob,
):

    threshold_values = np.arange(
        0.05,
        0.51,
        0.05,
    )

    rows = []

    for threshold in threshold_values:

        predictions = (
            valid_prob >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),

                "precision": precision_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),

                "recall": recall_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),

                "f1": f1_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),

                "f2": fbeta_score(
                    y_valid,
                    predictions,
                    beta=2,
                    zero_division=0,
                ),

                "predicted_positive": int(
                    predictions.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FINAL METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f2": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train(
    config: TrainConfig,
):

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_dir = Path(
        config.output_dir
    )

    models_dir = (
        output_dir / "models"
    )

    reports_dir = (
        output_dir / "reports"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    log.info(
        "Loading training and test data..."
    )

    train_df = pd.read_csv(
        config.train_file
    )

    test_df = pd.read_csv(
        config.test_file
    )

    y_train = (
        train_df[TARGET_COL]
        .astype(int)
    )

    y_test = (
        test_df[TARGET_COL]
        .astype(int)
    )

    log.info(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    log.info(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    log.info(
        f"Train positive rate: "
        f"{y_train.mean():.4%}"
    )

    log.info(
        f"Test positive rate: "
        f"{y_test.mean():.4%}"
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    X_train_full = (
        build_feature_matrix(
            train_df
        )
    )

    feature_columns = list(
        X_train_full.columns
    )

    X_test = build_feature_matrix(
        test_df,
        feature_columns=feature_columns,
    )

    log.info(
        f"Features used: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    X_fit, X_valid, y_fit, y_valid = (
        train_test_split(
            X_train_full,
            y_train,
            test_size=config.validation_size,
            stratify=y_train,
            random_state=config.random_state,
        )
    )

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_fit_imp = pd.DataFrame(
        imputer.fit_transform(
            X_fit
        ),
        columns=feature_columns,
        index=X_fit.index,
    )

    X_valid_imp = pd.DataFrame(
        imputer.transform(
            X_valid
        ),
        columns=feature_columns,
        index=X_valid.index,
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(
            X_test
        ),
        columns=feature_columns,
        index=X_test.index,
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    positive_rate = y_fit.mean()

    base_scale_pos_weight = max(
        1.0,
        (1 - positive_rate) /
        positive_rate,
    )

    log.info(
        f"Base scale_pos_weight: "
        f"{base_scale_pos_weight:.2f}"
    )

    # --------------------------------------------------------
    # Base XGBoost model
    # --------------------------------------------------------

    base_model = XGBClassifier(

        objective="binary:logistic",

        eval_metric="aucpr",

        tree_method="hist",

        random_state=config.random_state,

        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Hyperparameter search space
    # --------------------------------------------------------

    param_distributions = {

        "n_estimators": [
            200,
            300,
            400,
            600,
        ],

        "max_depth": [
            2,
            3,
            4,
            5,
        ],

        "learning_rate": [
            0.01,
            0.02,
            0.03,
            0.05,
            0.08,
        ],

        "min_child_weight": [
            1,
            3,
            5,
            10,
        ],

        "gamma": [
            0,
            0.05,
            0.1,
            0.25,
            0.5,
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "reg_alpha": [
            0,
            0.01,
            0.1,
            0.5,
            1,
        ],

        "reg_lambda": [
            1,
            3,
            5,
            10,
            20,
        ],

        "scale_pos_weight": sorted(
            {
                round(
                    base_scale_pos_weight * multiplier,
                    1,
                )

                for multiplier in [
                    0.5,
                    0.75,
                    1.0,
                    1.5,
                    2.0,
                ]
            }
        ),
    }

    # --------------------------------------------------------
    # Stratified CV
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    # --------------------------------------------------------
    # RandomizedSearchCV
    # --------------------------------------------------------

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

    log.info(
        "Starting hyperparameter search..."
    )

    log.info(
        "Scoring metric: PR-AUC / Average Precision"
    )

    search.fit(
        X_fit_imp,
        y_fit,
    )

    log.info(
        f"Best CV PR-AUC: "
        f"{search.best_score_:.4f}"
    )

    log.info(
        f"Best parameters: "
        f"{search.best_params_}"
    )

    best_params = (
        search.best_params_
    )

    # --------------------------------------------------------
    # Ensemble seeds
    # --------------------------------------------------------

    seeds = [
        config.random_state + i
        for i in range(
            config.n_ensemble_seeds
        )
    ]

    log.info(
        f"Using ensemble seeds: "
        f"{seeds}"
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    log.info(
        "Generating validation probabilities..."
    )

    valid_prob = ensemble_predict_proba(

        X_fit_imp,

        y_fit,

        X_valid_imp,

        best_params,

        seeds,
    )

    # --------------------------------------------------------
    # Select threshold
    # --------------------------------------------------------

    (
        selected_threshold,
        validation_precision,
        validation_recall,
    ) = select_threshold(

        y_valid,

        valid_prob,

        config.target_recall,
    )

    log.info(
        "=" * 70
    )

    log.info(
        "SELECTED THRESHOLD"
    )

    log.info(
        "=" * 70
    )

    log.info(
        f"Threshold: "
        f"{selected_threshold:.4f}"
    )

    log.info(
        f"Validation precision: "
        f"{validation_precision:.4f}"
    )

    log.info(
        f"Validation recall: "
        f"{validation_recall:.4f}"
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    threshold_table = (
        create_threshold_table(
            y_valid,
            valid_prob,
        )
    )

    threshold_table.to_csv(
        reports_dir /
        "threshold_analysis.csv",
        index=False,
    )

    log.info(
        "Threshold analysis saved."
    )

    # --------------------------------------------------------
    # Refit imputer on ALL training data
    # --------------------------------------------------------

    log.info(
        "Refitting imputer on full training data..."
    )

    X_train_full_imp = pd.DataFrame(
        imputer.fit_transform(
            X_train_full
        ),
        columns=feature_columns,
        index=X_train_full.index,
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(
            X_test
        ),
        columns=feature_columns,
        index=X_test.index,
    )

    # --------------------------------------------------------
    # Final test prediction
    # --------------------------------------------------------

    log.info(
        "Training final ensemble..."
    )

    test_prob = ensemble_predict_proba(

        X_train_full_imp,

        y_train,

        X_test_imp,

        best_params,

        seeds,
    )

    test_pred = (
        test_prob >= selected_threshold
    ).astype(int)

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(

        y_test,

        test_prob,

        selected_threshold,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_pred,
    )

    tn, fp, fn, tp = cm.ravel()

    metrics[
        "test_prevalence"
    ] = float(
        y_test.mean()
    )

    metrics[
        "test_positive_count"
    ] = int(
        y_test.sum()
    )

    metrics[
        "test_negative_count"
    ] = int(
        (y_test == 0).sum()
    )

    metrics[
        "cv_best_pr_auc"
    ] = float(
        search.best_score_
    )

    metrics[
        "best_params"
    ] = search.best_params_

    metrics[
        "confusion_matrix"
    ] = {

        "true_negative": int(tn),

        "false_positive": int(fp),

        "false_negative": int(fn),

        "true_positive": int(tp),
    }

    # --------------------------------------------------------
    # Final results logging
    # --------------------------------------------------------

    log.info(
        "=" * 70
    )

    log.info(
        "FINAL TEST RESULTS"
    )

    log.info(
        "=" * 70
    )

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
    ]:

        log.info(
            f"{metric:>20}: "
            f"{metrics[metric]:.4f}"
        )

    log.info(
        f"{'test_prevalence':>20}: "
        f"{metrics['test_prevalence']:.4%}"
    )

    log.info(
        f"Confusion Matrix: "
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )

    log.info(
        f"PR-AUC vs baseline: "
        f"{metrics['pr_auc']:.4f} "
        f"vs "
        f"{metrics['test_prevalence']:.4f}"
    )

    # --------------------------------------------------------
    # Weak signal warning
    # --------------------------------------------------------

    if metrics["roc_auc"] < 0.65:

        log.warning(
            "ROC-AUC is below 0.65."
        )

        log.warning(
            "Current features provide weak "
            "predictive signal for repeat purchase."
        )

        log.warning(
            "Future improvement should focus "
            "on richer customer/product/payment/"
            "review/geography features."
        )

    # ========================================================
    # SAVE FINAL MODELS
    # ========================================================

    log.info(
        "Saving final production models..."
    )

    final_models = []

    for seed in seeds:

        model = create_model(
            best_params,
            seed,
        )

        model.fit(
            X_train_full_imp,
            y_train,
        )

        final_models.append(
            model
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    joblib.dump(
        final_models,
        models_dir /
        "xgb_repeat_purchase_production.joblib",
    )

    # --------------------------------------------------------
    # Imputer
    # --------------------------------------------------------

    joblib.dump(
        imputer,
        models_dir /
        "imputer.joblib",
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    with open(
        models_dir /
        "feature_columns.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    with open(
        reports_dir /
        "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "config": asdict(
                    config
                ),

                "selected_threshold": (
                    selected_threshold
                ),

                "validation_precision": (
                    validation_precision
                ),

                "validation_recall": (
                    validation_recall
                ),

                "final_test_metrics": (
                    metrics
                ),
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Threshold config
    # --------------------------------------------------------

    with open(
        reports_dir /
        "threshold.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "selected_threshold": (
                    selected_threshold
                ),

                "target_recall": (
                    config.target_recall
                ),

                "threshold_source": (
                    "validation_set"
                ),
            },
            f,
            indent=2,
        )

    log.info(
        "=" * 70
    )

    log.info(
        "PRODUCTION TRAINING COMPLETE"
    )

    log.info(
        "=" * 70
    )

    log.info(
        f"Models saved to: "
        f"{models_dir.resolve()}"
    )

    log.info(
        f"Reports saved to: "
        f"{reports_dir.resolve()}"
    )

    return metrics


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--train-file",
        required=True,
    )

    parser.add_argument(
        "--test-file",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts_production",
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.40,
        help=(
            "Minimum recall target used "
            "for validation threshold selection."
        ),
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=40,
        help=(
            "RandomizedSearchCV iterations."
        ),
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--n-ensemble-seeds",
        type=int,
        default=3,
    )

    args = parser.parse_args(
        argv
    )

    if not (
        0 < args.target_recall < 1
    ):

        parser.error(
            "--target-recall must be "
            "between 0 and 1."
        )

    if args.cv_folds < 2:

        parser.error(
            "--cv-folds must be "
            "at least 2."
        )

    return TrainConfig(

        train_file=args.train_file,

        test_file=args.test_file,

        output_dir=args.output_dir,

        target_recall=args.target_recall,

        n_iter=args.n_iter,

        cv_folds=args.cv_folds,

        random_state=args.random_state,

        validation_size=args.validation_size,

        n_ensemble_seeds=(
            args.n_ensemble_seeds
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main(argv=None):

    config = parse_args(
        argv
    )

    try:

        train(config)

    except Exception:

        log.exception(
            "Training failed."
        )

        sys.exit(1)


if __name__ == "__main__":

    main()