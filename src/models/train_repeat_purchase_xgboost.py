"""
Production-level XGBoost training pipeline
for E-Commerce Customer Intelligence & Repeat Purchase Propensity System.

Includes:
- Feature engineering
- PR-AUC based hyperparameter tuning
- Stratified validation
- Precision-floor / Recall-floor threshold selection
- Configurable business threshold
- 5-fold stability analysis
- Approximate 95% CI from fold metrics
- Bootstrap 95% CI on untouched test metrics
- Multi-seed XGBoost ensemble
- Optional isotonic calibration
- Final untouched test evaluation
- Business threshold analysis
- Artifact saving
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

from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("repeat_purchase_pipeline")


# ============================================================================
# CONSTANTS
# ============================================================================

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


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # ------------------------------------------------------------------------
    # Missingness indicators
    # ------------------------------------------------------------------------

    for col in GAP_COLUMNS:

        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)

    # ------------------------------------------------------------------------
    # Safe denominators
    # ------------------------------------------------------------------------

    safe_orders = df["total_orders"].replace(0, np.nan)

    safe_lifetime = df["customer_lifetime_days"].replace(
        0,
        np.nan,
    )

    safe_window = df["observation_window_days"].replace(
        0,
        np.nan,
    )

    # ------------------------------------------------------------------------
    # Ratio / rate features
    # ------------------------------------------------------------------------

    df["revenue_per_order"] = (
        df["total_revenue"] / safe_orders
    )

    df["orders_per_lifetime_day"] = (
        df["total_orders"] / safe_lifetime
    )

    df["revenue_per_lifetime_day"] = (
        df["total_revenue"] / safe_lifetime
    )

    df["recency_ratio"] = (
        df["recency_days"] / safe_window
    )

    df["purchase_frequency_ratio"] = (
        df["purchase_frequency"] / safe_window
    )

    df["gap_ratio"] = (
        df["mean_purchase_gap_days"] / safe_window
    )

    df["revenue_order_frequency"] = (
        df["total_revenue"] /
        (df["total_orders"] + 1)
    )

    # ------------------------------------------------------------------------
    # Log transforms
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------------

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


# ============================================================================
# FEATURE MATRIX
# ============================================================================

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

    if feature_columns is not None:

        for col in feature_columns:

            if col not in X.columns:
                X[col] = np.nan

        X = X[feature_columns]

    return X


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TrainConfig:

    train_file: str

    test_file: str

    output_dir: str = "artifacts"

    optimization_mode: str = "recall_floor"

    target_recall: float = 0.40

    target_precision: float = 0.10

    n_iter: int = 40

    cv_folds: int = 5

    random_state: int = 42

    validation_size: float = 0.20

    calibrate: bool = True

    n_ensemble_seeds: int = 3

    # New
    threshold: Optional[float] = None

    # Business threshold table
    threshold_start: float = 0.05

    threshold_end: float = 0.50

    threshold_step: float = 0.05

    # Bootstrap
    bootstrap_iterations: int = 2000

    bootstrap_confidence: float = 0.95


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_xgb_model(
    params: dict,
    seed: int,
):

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        **params,
    )


# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================

def ensemble_predict_proba(
    fit_X,
    fit_y,
    predict_X,
    best_params,
    seeds,
    calibrate=True,
):

    probs = np.zeros(
        len(predict_X),
        dtype=float,
    )

    for seed in seeds:

        model = create_xgb_model(
            best_params,
            seed,
        )

        if calibrate:

            model = CalibratedClassifierCV(
                model,
                method="isotonic",
                cv=3,
            )

        model.fit(
            fit_X,
            fit_y,
        )

        probs += model.predict_proba(
            predict_X
        )[:, 1]

    return probs / len(seeds)


# ============================================================================
# BASIC METRICS
# ============================================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),

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


# ============================================================================
# THRESHOLD SELECTION
# ============================================================================

def select_threshold(
    y_valid,
    valid_prob,
    config: TrainConfig,
):

    precisions, recalls, thresholds = (
        precision_recall_curve(
            y_valid,
            valid_prob,
        )
    )

    # Align arrays
    precisions = precisions[:-1]

    recalls = recalls[:-1]

    if len(thresholds) == 0:

        return (
            0.50,
            0.0,
            0.0,
        )

    # ------------------------------------------------------------------------
    # Manual threshold has highest priority
    # ------------------------------------------------------------------------

    if config.threshold is not None:

        threshold = float(
            config.threshold
        )

        predictions = (
            valid_prob >= threshold
        ).astype(int)

        precision = precision_score(
            y_valid,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_valid,
            predictions,
            zero_division=0,
        )

        log.info(
            f"Manual business threshold selected: "
            f"{threshold:.4f}"
        )

        log.info(
            f"Validation precision={precision:.4f}, "
            f"recall={recall:.4f}"
        )

        return (
            threshold,
            precision,
            recall,
        )

    # ------------------------------------------------------------------------
    # Precision floor
    # ------------------------------------------------------------------------

    if config.optimization_mode == "precision_floor":

        candidates = [

            (t, p, r)

            for t, p, r in zip(
                thresholds,
                precisions,
                recalls,
            )

            if p >= config.target_precision

        ]

        if candidates:

            best_threshold, best_precision, best_recall = max(
                candidates,
                key=lambda x: x[2],
            )

        else:

            log.warning(
                f"No validation threshold reached "
                f"target precision "
                f"{config.target_precision:.2%}."
            )

            best_idx = int(
                np.argmax(precisions)
            )

            best_threshold = thresholds[
                best_idx
            ]

            best_precision = precisions[
                best_idx
            ]

            best_recall = recalls[
                best_idx
            ]

    # ------------------------------------------------------------------------
    # Recall floor
    # ------------------------------------------------------------------------

    else:

        candidates = [

            (t, p, r)

            for t, p, r in zip(
                thresholds,
                precisions,
                recalls,
            )

            if r >= config.target_recall

        ]

        if candidates:

            best_threshold, best_precision, best_recall = max(
                candidates,
                key=lambda x: x[1],
            )

        else:

            log.warning(
                f"No validation threshold reached "
                f"target recall "
                f"{config.target_recall:.2%}."
            )

            f2_scores = [

                fbeta_score(
                    y_valid,
                    (
                        valid_prob >= t
                    ).astype(int),
                    beta=2,
                    zero_division=0,
                )

                for t in thresholds
            ]

            best_idx = int(
                np.argmax(f2_scores)
            )

            best_threshold = thresholds[
                best_idx
            ]

            best_precision = precisions[
                best_idx
            ]

            best_recall = recalls[
                best_idx
            ]

    return (
        float(best_threshold),
        float(best_precision),
        float(best_recall),
    )


# ============================================================================
# BUSINESS THRESHOLD TABLE
# ============================================================================

def create_threshold_analysis(
    y_true,
    probabilities,
    config: TrainConfig,
):

    thresholds = np.arange(
        config.threshold_start,
        config.threshold_end + (
            config.threshold_step / 2
        ),
        config.threshold_step,
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        predicted_positive = int(
            predictions.sum()
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f2 = fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": round(
                    float(threshold),
                    4,
                ),

                "precision": precision,

                "recall": recall,

                "f1": f1,

                "f2": f2,

                "predicted_positive": predicted_positive,

                "predicted_positive_rate": (
                    predicted_positive /
                    len(y_true)
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# 5-FOLD STABILITY ANALYSIS
# ============================================================================

def run_stability_analysis(
    X,
    y,
    threshold,
    best_params,
    config: TrainConfig,
):

    log.info("=" * 70)

    log.info(
        "5-FOLD STABILITY ANALYSIS"
    )

    log.info("=" * 70)

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    fold_results = []

    # We use one seed for stability analysis
    # to keep computation reasonable.
    stability_seed = config.random_state

    for fold_number, (
        train_idx,
        valid_idx,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        log.info(
            f"Stability fold {fold_number}/"
            f"{config.cv_folds}"
        )

        X_fold_train = X.iloc[
            train_idx
        ]

        X_fold_valid = X.iloc[
            valid_idx
        ]

        y_fold_train = y.iloc[
            train_idx
        ]

        y_fold_valid = y.iloc[
            valid_idx
        ]

        fold_imputer = SimpleImputer(
            strategy="median"
        )

        X_fold_train_imp = pd.DataFrame(
            fold_imputer.fit_transform(
                X_fold_train
            ),
            columns=X.columns,
            index=X_fold_train.index,
        )

        X_fold_valid_imp = pd.DataFrame(
            fold_imputer.transform(
                X_fold_valid
            ),
            columns=X.columns,
            index=X_fold_valid.index,
        )

        model = create_xgb_model(
            best_params,
            stability_seed,
        )

        if config.calibrate:

            model = CalibratedClassifierCV(
                model,
                method="isotonic",
                cv=3,
            )

        model.fit(
            X_fold_train_imp,
            y_fold_train,
        )

        fold_prob = model.predict_proba(
            X_fold_valid_imp
        )[:, 1]

        fold_metrics = calculate_metrics(
            y_fold_valid,
            fold_prob,
            threshold,
        )

        fold_metrics[
            "fold"
        ] = fold_number

        fold_metrics[
            "positive_rate"
        ] = float(
            y_fold_valid.mean()
        )

        fold_results.append(
            fold_metrics
        )

    fold_df = pd.DataFrame(
        fold_results
    )

    metric_names = [
        "precision",
        "recall",
        "f1",
        "f2",
        "pr_auc",
        "roc_auc",
    ]

    summary = {}

    for metric in metric_names:

        values = fold_df[
            metric
        ].astype(float).values

        mean_value = float(
            np.mean(values)
        )

        std_value = float(
            np.std(
                values,
                ddof=1,
            )
        )

        n = len(values)

        # Approximate 95% CI
        margin = (
            1.96 *
            std_value /
            np.sqrt(n)
        )

        summary[metric] = {
            "mean": mean_value,
            "std": std_value,
            "ci_95_lower": max(
                0.0,
                mean_value - margin,
            ),
            "ci_95_upper": min(
                1.0,
                mean_value + margin,
            ),
        }

        log.info(
            f"{metric:>10}: "
            f"{mean_value:.4f} "
            f"+/- {std_value:.4f}"
        )

    return {
        "fold_results": fold_results,
        "summary": summary,
        "threshold_used": float(threshold),
    }


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================================

def bootstrap_metric_ci(
    y_true,
    probabilities,
    threshold,
    metric_name,
    iterations=2000,
    confidence=0.95,
    random_state=42,
):

    rng = np.random.default_rng(
        random_state
    )

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities
    )

    n = len(y_true)

    values = []

    for _ in range(iterations):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample_y = y_true[
            indices
        ]

        sample_prob = probabilities[
            indices
        ]

        # Skip degenerate bootstrap samples
        if (
            len(np.unique(sample_y))
            < 2
        ):
            continue

        predictions = (
            sample_prob >= threshold
        ).astype(int)

        if metric_name == "precision":

            value = precision_score(
                sample_y,
                predictions,
                zero_division=0,
            )

        elif metric_name == "recall":

            value = recall_score(
                sample_y,
                predictions,
                zero_division=0,
            )

        elif metric_name == "f1":

            value = f1_score(
                sample_y,
                predictions,
                zero_division=0,
            )

        elif metric_name == "f2":

            value = fbeta_score(
                sample_y,
                predictions,
                beta=2,
                zero_division=0,
            )

        elif metric_name == "pr_auc":

            value = average_precision_score(
                sample_y,
                sample_prob,
            )

        elif metric_name == "roc_auc":

            value = roc_auc_score(
                sample_y,
                sample_prob,
            )

        else:

            raise ValueError(
                f"Unknown metric: {metric_name}"
            )

        values.append(
            float(value)
        )

    values = np.asarray(
        values
    )

    alpha = (
        1 -
        confidence
    )

    lower = np.quantile(
        values,
        alpha / 2,
    )

    upper = np.quantile(
        values,
        1 - alpha / 2,
    )

    return {
        "estimate": float(
            calculate_metrics(
                y_true,
                probabilities,
                threshold,
            )[metric_name]
        ),
        "ci_lower": float(
            lower
        ),
        "ci_upper": float(
            upper
        ),
        "confidence": confidence,
        "iterations": len(values),
    }


def run_bootstrap_analysis(
    y_test,
    test_prob,
    threshold,
    config: TrainConfig,
):

    log.info("=" * 70)

    log.info(
        "BOOTSTRAP 95% CONFIDENCE INTERVAL"
    )

    log.info("=" * 70)

    metrics = [
        "precision",
        "recall",
        "f1",
        "f2",
        "pr_auc",
        "roc_auc",
    ]

    results = {}

    for metric in metrics:

        results[metric] = bootstrap_metric_ci(
            y_test,
            test_prob,
            threshold,
            metric,
            iterations=config.bootstrap_iterations,
            confidence=config.bootstrap_confidence,
            random_state=(
                config.random_state + 100
            ),
        )

        result = results[metric]

        log.info(
            f"{metric:>10}: "
            f"{result['estimate']:.4f} "
            f"["
            f"{result['ci_lower']:.4f}, "
            f"{result['ci_upper']:.4f}"
            f"]"
        )

    return results


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train(
    config: TrainConfig,
):

    out_dir = Path(
        config.output_dir
    )

    models_dir = (
        out_dir / "models"
    )

    reports_dir = (
        out_dir / "reports"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    log.info(
        "Loading data ..."
    )

    train_df = pd.read_csv(
        config.train_file
    )

    test_df = pd.read_csv(
        config.test_file
    )

    y_train_full = (
        train_df[TARGET_COL]
        .astype(int)
    )

    y_test = (
        test_df[TARGET_COL]
        .astype(int)
    )

    # ------------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------------

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
        f"Features used "
        f"({len(feature_columns)}): "
        f"{feature_columns}"
    )

    log.info(
        f"Train positive rate: "
        f"{y_train_full.mean():.4%}"
    )

    log.info(
        f"Test positive rate: "
        f"{y_test.mean():.4%}"
    )

    # ------------------------------------------------------------------------
    # VALIDATION SPLIT
    # ------------------------------------------------------------------------

    X_fit, X_valid, y_fit, y_valid = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=config.validation_size,
            stratify=y_train_full,
            random_state=config.random_state,
        )
    )

    # ------------------------------------------------------------------------
    # IMPUTATION
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # HYPERPARAMETER SEARCH
    # ------------------------------------------------------------------------

    pos_rate = y_fit.mean()

    base_spw = max(
        1.0,
        (1 - pos_rate) / pos_rate,
    )

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=-1,
    )

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
                    base_spw * multiplier,
                    1,
                )

                for multiplier in (
                    0.5,
                    0.75,
                    1.0,
                    1.5,
                    2.0,
                )
            }
        ),
    }

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

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
        "Starting hyperparameter search "
        "(scoring = PR-AUC) ..."
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
        f"Best params: "
        f"{search.best_params_}"
    )

    best_params = search.best_params_

    # ------------------------------------------------------------------------
    # ENSEMBLE VALIDATION PREDICTION
    # ------------------------------------------------------------------------

    seeds = [
        config.random_state + i
        for i in range(
            config.n_ensemble_seeds
        )
    ]

    log.info(
        f"Training "
        f"{len(seeds)}-seed ensemble "
        f"for validation ..."
    )

    valid_prob = ensemble_predict_proba(
        X_fit_imp,
        y_fit,
        X_valid_imp,
        best_params,
        seeds,
        config.calibrate,
    )

    # ------------------------------------------------------------------------
    # THRESHOLD SELECTION
    # ------------------------------------------------------------------------

    (
        best_threshold,
        best_precision,
        best_recall,
    ) = select_threshold(
        y_valid,
        valid_prob,
        config,
    )

    log.info(
        "=" * 70
    )

    log.info(
        "SELECTED BUSINESS THRESHOLD"
    )

    log.info(
        "=" * 70
    )

    log.info(
        f"Threshold: "
        f"{best_threshold:.4f}"
    )

    log.info(
        f"Validation precision: "
        f"{best_precision:.4f}"
    )

    log.info(
        f"Validation recall: "
        f"{best_recall:.4f}"
    )

    # ------------------------------------------------------------------------
    # 5-FOLD STABILITY
    #
    # This is separate from RandomizedSearchCV.
    # It measures how stable the selected operating point is.
    # ------------------------------------------------------------------------

    stability_report = run_stability_analysis(
        X_train_full,
        y_train_full,
        best_threshold,
        best_params,
        config,
    )

    # ------------------------------------------------------------------------
    # REFIT ON ALL TRAINING DATA
    # ------------------------------------------------------------------------

    log.info(
        "=" * 70
    )

    log.info(
        "REFITTING FINAL ENSEMBLE"
    )

    log.info(
        "=" * 70
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

    test_prob = ensemble_predict_proba(
        X_train_full_imp,
        y_train_full,
        X_test_imp,
        best_params,
        seeds,
        config.calibrate,
    )

    # ------------------------------------------------------------------------
    # FINAL TEST
    # ------------------------------------------------------------------------

    final_metrics = calculate_metrics(
        y_test,
        test_prob,
        best_threshold,
    )

    # ------------------------------------------------------------------------
    # BUSINESS THRESHOLD ANALYSIS
    #
    # IMPORTANT:
    # We use VALIDATION probabilities here,
    # NOT test probabilities, so test remains untouched.
    # ------------------------------------------------------------------------

    threshold_table = (
        create_threshold_analysis(
            y_valid,
            valid_prob,
            config,
        )
    )

    threshold_table_path = (
        reports_dir /
        "threshold_analysis.csv"
    )

    threshold_table.to_csv(
        threshold_table_path,
        index=False,
    )

    log.info(
        "=" * 70
    )

    log.info(
        "BUSINESS THRESHOLD ANALYSIS"
    )

    log.info(
        "=" * 70
    )

    print(
        threshold_table.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "f2": "{:.4f}".format,
                "predicted_positive_rate": "{:.4%}".format,
            },
        )
    )

    # ------------------------------------------------------------------------
    # BOOTSTRAP CI
    #
    # This is ONLY confidence estimation.
    # It does NOT change threshold or model.
    # ------------------------------------------------------------------------

    bootstrap_report = (
        run_bootstrap_analysis(
            y_test,
            test_prob,
            best_threshold,
            config,
        )
    )

    # ------------------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------------------

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        test_pred,
    )

    tn, fp, fn, tp = cm.ravel()

    final_metrics[
        "test_prevalence"
    ] = float(
        y_test.mean()
    )

    final_metrics[
        "cv_best_pr_auc"
    ] = float(
        search.best_score_
    )

    final_metrics[
        "best_params"
    ] = search.best_params_

    final_metrics[
        "confusion_matrix"
    ] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    final_metrics[
        "test_positive_count"
    ] = int(
        y_test.sum()
    )

    final_metrics[
        "test_negative_count"
    ] = int(
        (y_test == 0).sum()
    )

    # ------------------------------------------------------------------------
    # FINAL LOGGING
    # ------------------------------------------------------------------------

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
            f"{final_metrics[metric]:.4f}"
        )

    log.info(
        f"Test prevalence: "
        f"{final_metrics['test_prevalence']:.4%}"
    )

    log.info(
        f"PR-AUC vs baseline: "
        f"{final_metrics['pr_auc']:.4f} "
        f"vs "
        f"{final_metrics['test_prevalence']:.4f}"
    )

    log.info(
        f"Confusion Matrix: "
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )

    # ------------------------------------------------------------------------
    # WEAK SIGNAL WARNING
    # ------------------------------------------------------------------------

    if final_metrics["roc_auc"] < 0.65:

        log.warning(
            "ROC-AUC is still low (<0.65). "
            "Current features carry weak signal "
            "for repeat purchase prediction."
        )

        log.warning(
            "Consider adding product/category, "
            "payment, geography, seller and "
            "review-score features."
        )

    # ------------------------------------------------------------------------
    # SAVE FINAL MODELS
    # ------------------------------------------------------------------------

    log.info(
        "Saving final ensemble models ..."
    )

    final_models = []

    for seed in seeds:

        model = create_xgb_model(
            best_params,
            seed,
        )

        if config.calibrate:

            model = CalibratedClassifierCV(
                model,
                method="isotonic",
                cv=3,
            )

        model.fit(
            X_train_full_imp,
            y_train_full,
        )

        final_models.append(
            model
        )

    # ------------------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------------------

    joblib.dump(
        final_models,
        models_dir /
        "xgb_repeat_purchase_ensemble.joblib",
    )

    joblib.dump(
        imputer,
        models_dir /
        "imputer.joblib",
    )

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

    # ------------------------------------------------------------------------
    # SAVE MAIN METRICS
    # ------------------------------------------------------------------------

    with open(
        reports_dir /
        "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "config": asdict(config),

                "selected_threshold": (
                    best_threshold
                ),

                "validation_precision": (
                    best_precision
                ),

                "validation_recall": (
                    best_recall
                ),

                "final_test_metrics": (
                    final_metrics
                ),
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # SAVE STABILITY REPORT
    # ------------------------------------------------------------------------

    with open(
        reports_dir /
        "stability_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stability_report,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # SAVE BOOTSTRAP REPORT
    # ------------------------------------------------------------------------

    with open(
        reports_dir /
        "bootstrap_ci.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            bootstrap_report,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # SAVE THRESHOLD CONFIG
    # ------------------------------------------------------------------------

    threshold_config = {

        "selected_threshold": (
            best_threshold
        ),

        "optimization_mode": (
            config.optimization_mode
        ),

        "target_precision": (
            config.target_precision
        ),

        "target_recall": (
            config.target_recall
        ),

        "threshold_source": (
            "manual"
            if config.threshold is not None
            else "validation"
        ),
    }

    with open(
        reports_dir /
        "threshold_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            threshold_config,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # SAVE COMPLETE RUN SUMMARY
    # ------------------------------------------------------------------------

    run_summary = {

        "selected_threshold": (
            best_threshold
        ),

        "validation": {

            "precision": (
                best_precision
            ),

            "recall": (
                best_recall
            ),
        },

        "test": final_metrics,

        "stability": stability_report,

        "bootstrap_ci": bootstrap_report,

        "threshold_analysis_file": str(
            threshold_table_path
        ),

        "models": str(
            models_dir
        ),
    }

    with open(
        reports_dir /
        "run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # FINAL MESSAGE
    # ------------------------------------------------------------------------

    log.info(
        "=" * 70
    )

    log.info(
        "TRAINING COMPLETE"
    )

    log.info(
        "=" * 70
    )

    log.info(
        f"Artifacts saved to: "
        f"{out_dir.resolve()}"
    )

    log.info(
        f"Threshold analysis: "
        f"{threshold_table_path}"
    )

    log.info(
        f"Stability report: "
        f"{reports_dir / 'stability_report.json'}"
    )

    log.info(
        f"Bootstrap CI: "
        f"{reports_dir / 'bootstrap_ci.json'}"
    )

    return run_summary


# ============================================================================
# CLI
# ============================================================================

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
        default="artifacts",
    )

    parser.add_argument(
        "--optimization-mode",
        choices=[
            "recall_floor",
            "precision_floor",
        ],
        default="precision_floor",
        help=(
            "recall_floor: maximize precision "
            "subject to minimum recall. "
            "precision_floor: maximize recall "
            "subject to minimum precision."
        ),
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.40,
        help=(
            "Minimum recall for recall_floor mode."
        ),
    )

    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.10,
        help=(
            "Minimum precision for precision_floor mode."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Manual business threshold. "
            "If provided, automatic threshold "
            "selection is skipped."
        ),
    )

    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--threshold-end",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
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
        "--no-calibrate",
        action="store_true",
        help=(
            "Disable isotonic probability calibration."
        ),
    )

    parser.add_argument(
        "--n-ensemble-seeds",
        type=int,
        default=3,
        help=(
            "Number of XGBoost seeds to average."
        ),
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help=(
            "Number of bootstrap samples "
            "for confidence intervals."
        ),
    )

    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help=(
            "Bootstrap confidence level."
        ),
    )

    args = parser.parse_args(
        argv
    )

    # ------------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------------

    if args.threshold is not None:

        if not (
            0 < args.threshold < 1
        ):

            parser.error(
                "--threshold must be between 0 and 1"
            )

    if not (
        0 < args.target_precision < 1
    ):

        parser.error(
            "--target-precision must be "
            "between 0 and 1"
        )

    if not (
        0 < args.target_recall < 1
    ):

        parser.error(
            "--target-recall must be "
            "between 0 and 1"
        )

    if args.cv_folds < 2:

        parser.error(
            "--cv-folds must be at least 2"
        )

    if args.bootstrap_iterations < 100:

        parser.error(
            "--bootstrap-iterations should "
            "be at least 100"
        )

    return TrainConfig(

        train_file=args.train_file,

        test_file=args.test_file,

        output_dir=args.output_dir,

        optimization_mode=args.optimization_mode,

        target_recall=args.target_recall,

        target_precision=args.target_precision,

        threshold=args.threshold,

        threshold_start=args.threshold_start,

        threshold_end=args.threshold_end,

        threshold_step=args.threshold_step,

        n_iter=args.n_iter,

        cv_folds=args.cv_folds,

        random_state=args.random_state,

        validation_size=args.validation_size,

        calibrate=not args.no_calibrate,

        n_ensemble_seeds=args.n_ensemble_seeds,

        bootstrap_iterations=args.bootstrap_iterations,

        bootstrap_confidence=args.bootstrap_confidence,
    )


# ============================================================================
# MAIN
# ============================================================================

def main(argv=None):

    config = parse_args(
        argv
    )

    try:

        train(config)

    except Exception:

        log.exception(
            "Training failed"
        )

        sys.exit(1)


if __name__ == "__main__":

    main()