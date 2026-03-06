"""
model/lgbm_model.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    The two LightGBM model classes plus Optuna tuning. Handles training,
    early stopping, cross-validation, threshold optimisation, and
    save / load of trained models as pickle files.

CALLED FROM
    model/train.py           SatelliteRiskRegressor, SatelliteRiskClassifier,
                              tune_hyperparameters
    model/inference.py       SatelliteRiskRegressor.load(), .load()
    model/lgbm_engine.py     same
    pipeline/model_bridge.py same

CALLS INTO
    lightgbm    lgb.Dataset, lgb.train, lgb.Booster
    optuna      create_study, trial.suggest_*
    sklearn     KFold, StratifiedKFold, metrics
    model/config.py  LGBM_REG_PARAMS, LGBM_CLF_PARAMS, EARLY_STOPPING_ROUNDS
    Nothing else from this project.

WHAT IT PROVIDES
    build_lgb_datasets(X_train, y_train, X_val, y_val, cat_cols)
        Returns (lgb.Dataset train, lgb.Dataset val).

    SatelliteRiskRegressor
        train(X_train, y_train, X_val, y_val)  trains, stores best_iteration
        cross_validate(X, y, n_folds=5) → list[float]  R² per fold
        predict(X) → np.ndarray  log10(Pc) risk scores
        save(path) / load(path)

    SatelliteRiskClassifier
        train(X_train, y_train, X_val, y_val)
        optimise_threshold(X_val, y_val)  finds F1-max threshold
        predict(X) → np.ndarray  binary 0/1
        predict_proba(X) → np.ndarray  probabilities [0,1]
        optimal_threshold  float (default 0.65)
        save(path) / load(path)

    tune_hyperparameters(X_train, y_train, X_val, y_val, task)
        Bayesian search over OPTUNA_SEARCH_SPACE.
        task = "regression" | "classification"
        Returns best params dict.

RENAMED FROM
    satellite_lgbm/model.py  →  model/lgbm_model.py
    (avoids shadowing Python's built-in "model" name)
══════════════════════════════════════════════════════════════════════════════
"""
import os
import logging
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score,
)

import config

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTGBM DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_lgb_datasets(
    X_train, y_train, X_val, y_val,
    cat_cols: Optional[List[str]] = None
) -> Tuple[lgb.Dataset, lgb.Dataset]:
    """Build LightGBM Dataset objects with categorical feature registration."""
    cat_cols = cat_cols or []
    # Only include categoricals that exist in X_train
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_cols if cat_cols else "auto",
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val, label=y_val,
        reference=dtrain,
        categorical_feature=cat_cols if cat_cols else "auto",
        free_raw_data=False,
    )
    return dtrain, dval


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SatelliteRiskRegressor:
    """
    LightGBM regressor for continuous collision risk prediction.
    Target: `risk` (log-scale collision probability)
    """

    def __init__(self, params: Dict = None):
        self.params   = params or config.LGBM_REG_PARAMS.copy()
        self.model    = None
        self.feature_names: List[str] = []
        self.best_iteration: int = 0
        self.val_metrics: Dict = {}

    def train(
        self,
        X_train, y_train,
        X_val,   y_val,
        feature_names: List[str] = None,
        cat_cols: List[str] = None,
    ) -> "SatelliteRiskRegressor":

        self.feature_names = feature_names or list(X_train.columns)
        logger.info(f"[Regression] Training on {len(X_train)} rows, "
                    f"{len(self.feature_names)} features")

        dtrain, dval = build_lgb_datasets(X_train, y_train, X_val, y_val, cat_cols)

        callbacks = [
            lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        self.best_iteration = self.model.best_iteration
        logger.info(f"[Regression] Best iteration: {self.best_iteration}")

        # Validation metrics
        y_pred = self.model.predict(X_val, num_iteration=self.best_iteration)
        self.val_metrics = {
            "MAE":  mean_absolute_error(y_val, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)),
            "R2":   r2_score(y_val, y_pred),
        }
        logger.info(f"[Regression] Val metrics: {self.val_metrics}")
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.best_iteration)

    def cross_validate(
        self, X, y, n_folds: int = 5, cat_cols: List[str] = None
    ) -> Dict:
        """K-Fold cross-validation for unbiased performance estimates."""
        kf     = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
        scores = {"MAE": [], "RMSE": [], "R2": []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            m = SatelliteRiskRegressor(self.params.copy())
            m.train(X_tr, y_tr, X_va, y_va, cat_cols=cat_cols)
            y_pred = m.predict(X_va)

            scores["MAE"].append(mean_absolute_error(y_va, y_pred))
            scores["RMSE"].append(np.sqrt(mean_squared_error(y_va, y_pred)))
            scores["R2"].append(r2_score(y_va, y_pred))
            logger.info(f"  Fold {fold+1}/{n_folds} — R²: {scores['R2'][-1]:.4f}")

        summary = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in scores.items()}
        logger.info(f"[Regression CV] R²: {summary['R2']['mean']:.4f} ± {summary['R2']['std']:.4f}")
        return summary

    def save(self, path: str = None):
        path = path or os.path.join(config.MODEL_DIR, "regressor.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Regressor saved → {path}")

    @classmethod
    def load(cls, path: str = None) -> "SatelliteRiskRegressor":
        path = path or os.path.join(config.MODEL_DIR, "regressor.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SatelliteRiskClassifier:
    """
    LightGBM binary classifier for high-risk collision alerts.
    Target: `high_risk` (1 = top 20% risk events)

    Includes threshold optimisation to maximise F1 score.
    """

    def __init__(self, params: Dict = None):
        self.params            = params or config.LGBM_CLF_PARAMS.copy()
        self.model             = None
        self.feature_names:    List[str] = []
        self.best_iteration:   int       = 0
        self.optimal_threshold: float    = 0.5
        self.val_metrics:      Dict      = {}

    def train(
        self,
        X_train, y_train,
        X_val,   y_val,
        feature_names: List[str] = None,
        cat_cols: List[str] = None,
    ) -> "SatelliteRiskClassifier":

        self.feature_names = feature_names or list(X_train.columns)
        pos_rate = y_train.mean()
        logger.info(f"[Classifier] Training on {len(X_train)} rows | "
                    f"Positive rate: {pos_rate*100:.1f}%")

        dtrain, dval = build_lgb_datasets(X_train, y_train, X_val, y_val, cat_cols)

        callbacks = [
            lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        self.best_iteration = self.model.best_iteration

        # Threshold optimisation on validation set
        y_proba = self.model.predict(X_val, num_iteration=self.best_iteration)
        self.optimal_threshold = self._find_optimal_threshold(y_val, y_proba)

        y_pred = (y_proba >= self.optimal_threshold).astype(int)
        self.val_metrics = {
            "ROC_AUC":         roc_auc_score(y_val, y_proba),
            "PR_AUC":          average_precision_score(y_val, y_proba),
            "F1":              f1_score(y_val, y_pred),
            "Precision":       precision_score(y_val, y_pred),
            "Recall":          recall_score(y_val, y_pred),
            "Threshold":       self.optimal_threshold,
        }
        logger.info(f"[Classifier] Best iteration: {self.best_iteration}")
        logger.info(f"[Classifier] Val metrics: {self.val_metrics}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.best_iteration)

    def predict(self, X, threshold: float = None) -> np.ndarray:
        threshold = threshold or self.optimal_threshold
        return (self.predict_proba(X) >= threshold).astype(int)

    @staticmethod
    def _find_optimal_threshold(y_true, y_proba) -> float:
        """Grid-search threshold that maximises F1 on validation set."""
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.91, 0.01):
            y_pred = (y_proba >= thresh).astype(int)
            if y_pred.sum() == 0:
                continue
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        logger.info(f"Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
        return float(best_thresh)

    def cross_validate(
        self, X, y, n_folds: int = 5, cat_cols: List[str] = None
    ) -> Dict:
        """Stratified K-Fold cross-validation."""
        skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
        scores = {"ROC_AUC": [], "F1": [], "PR_AUC": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            m = SatelliteRiskClassifier(self.params.copy())
            m.train(X_tr, y_tr, X_va, y_va, cat_cols=cat_cols)
            y_proba = m.predict_proba(X_va)

            scores["ROC_AUC"].append(roc_auc_score(y_va, y_proba))
            scores["F1"].append(f1_score(y_va, (y_proba >= m.optimal_threshold).astype(int)))
            scores["PR_AUC"].append(average_precision_score(y_va, y_proba))
            logger.info(f"  Fold {fold+1}/{n_folds} — AUC: {scores['ROC_AUC'][-1]:.4f}")

        summary = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in scores.items()}
        logger.info(f"[Classifier CV] AUC: {summary['ROC_AUC']['mean']:.4f} ± {summary['ROC_AUC']['std']:.4f}")
        return summary

    def save(self, path: str = None):
        path = path or os.path.join(config.MODEL_DIR, "classifier.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Classifier saved → {path}")

    @classmethod
    def load(cls, path: str = None) -> "SatelliteRiskClassifier":
        path = path or os.path.join(config.MODEL_DIR, "classifier.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_hyperparameters(
    task: str,          # "regression" or "classification"
    X_train, y_train,
    X_val,   y_val,
    cat_cols: List[str] = None,
    n_trials: int = config.OPTUNA_N_TRIALS,
    timeout:  int = config.OPTUNA_TIMEOUT_SEC,
) -> Dict:
    """
    Optuna Bayesian hyperparameter search.

    Parameters
    ----------
    task       : "regression" or "classification"
    n_trials   : number of Optuna trials
    timeout    : max seconds to run

    Returns
    -------
    best_params : dict of best hyperparameters
    """
    assert task in ("regression", "classification")
    sb = config.OPTUNA_SEARCH_SPACE
    cat_cols = [c for c in (cat_cols or []) if c in X_train.columns]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "regression" if task == "regression" else "binary",
            "metric":            "rmse"       if task == "regression" else "auc",
            "boosting_type":     "gbdt",
            "n_jobs":            -1,
            "random_state":      config.RANDOM_STATE,
            "verbose":           -1,
            "num_leaves":        trial.suggest_int("num_leaves",         *sb["num_leaves"]),
            "min_child_samples": trial.suggest_int("min_child_samples",  *sb["min_child_samples"]),
            "learning_rate":     trial.suggest_float("learning_rate",    *sb["learning_rate"],    log=True),
            "feature_fraction":  trial.suggest_float("feature_fraction", *sb["feature_fraction"]),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", *sb["bagging_fraction"]),
            "bagging_freq":      trial.suggest_int("bagging_freq",       *sb["bagging_freq"]),
            "reg_alpha":         trial.suggest_float("reg_alpha",        *sb["reg_alpha"]),
            "reg_lambda":        trial.suggest_float("reg_lambda",       *sb["reg_lambda"]),
            "n_estimators":      trial.suggest_int("n_estimators",       *sb["n_estimators"]),
        }
        if task == "classification":
            params["is_unbalance"] = True

        dtrain, dval = build_lgb_datasets(X_train, y_train, X_val, y_val, cat_cols)

        callbacks = [
            lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        model = lgb.train(
            params, dtrain,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        if task == "regression":
            return np.sqrt(mean_squared_error(y_val, y_pred))  # minimise RMSE
        else:
            return -roc_auc_score(y_val, y_pred)               # maximise AUC

    direction = "minimize"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    logger.info(f"[Optuna] Starting {n_trials}-trial search for {task}...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    logger.info(f"[Optuna] Best params: {best}")
    logger.info(f"[Optuna] Best value: {study.best_value:.6f}")

    return best, study
