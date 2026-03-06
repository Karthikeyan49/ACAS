"""
model/inference.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    High-level inference interface. Wraps both trained LightGBM models
    behind a clean API for single-event prediction from raw CDM feature
    dicts, and batch prediction from CSV files.

CALLED FROM
    pipeline/model_bridge.py   SatelliteCollisionPredictor().predict_single()
    model/train.py             run_inference() for --infer CLI mode

CALLS INTO
    model/config.py          MODEL_DIR
    data/data_pipeline.py    impute_missing, clip_outliers,
                              engineer_features, encode_categoricals
    model/lgbm_model.py      SatelliteRiskRegressor.load(),
                              SatelliteRiskClassifier.load()
    pandas, numpy

WHAT IT PROVIDES
    SatelliteCollisionPredictor(reg_path, clf_path, encoders)
        predict_single(feature_dict) → dict
            Input:  raw CDM feature dict (103 columns, NaN for missing)
            Output: {
                risk_score            float  log10(Pc)
                collision_probability float  linear Pc
                high_risk_probability float  classifier confidence 0-1
                is_high_risk          bool
                alert                 str    GREEN | YELLOW | ORANGE | RED
            }
        predict_batch(csv_path, output_path) → pd.DataFrame
        health_check() → dict

IMPORT CHANGES FROM ORIGINAL (satellite_lgbm/inference.py)
    import config               →  from model import config
    from data_pipeline import … →  from data.data_pipeline import …
    from model import …         →  from model.lgbm_model import …
══════════════════════════════════════════════════════════════════════════════
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from model import config
from data.data_pipeline import (impute_missing, clip_outliers, engineer_features,
                                 encode_categoricals, missing_value_report)
from model.lgbm_model import SatelliteRiskRegressor, SatelliteRiskClassifier

logger = logging.getLogger(__name__)


class SatelliteCollisionPredictor:
    """
    Unified inference interface.
    Loads trained regressor + classifier and applies full preprocessing pipeline.
    """

    def __init__(
        self,
        reg_path:  str = None,
        clf_path:  str = None,
        encoders:  Dict = None,
    ):
        self.regressor  = SatelliteRiskRegressor.load(reg_path)
        self.classifier = SatelliteRiskClassifier.load(clf_path)
        self.encoders   = encoders   # LabelEncoders fitted during training
        logger.info("Models loaded successfully.")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same preprocessing as training, using fitted encoders."""
        df = impute_missing(df)
        df = clip_outliers(df)
        df = engineer_features(df)
        df, _ = encode_categoricals(df, encoders=self.encoders, fit=False)
        return df

    def _align_features(self, df: pd.DataFrame, feature_names) -> pd.DataFrame:
        """Ensure columns match training feature set exactly."""
        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} features — filling with 0: {missing_cols[:5]}...")
            for col in missing_cols:
                df[col] = 0.0
        return df[feature_names]

    def predict_single(self, event: Dict) -> Dict:
        """
        Predict collision risk for a single conjunction event.

        Parameters
        ----------
        event : dict of feature values (raw, unprocessed)

        Returns
        -------
        dict with risk_score, collision_probability, alert_probability, alert_label
        """
        df = pd.DataFrame([event])
        df = self._preprocess(df)
        X  = self._align_features(df, self.regressor.feature_names)

        risk_score    = float(self.regressor.predict(X)[0])
        alert_proba   = float(self.classifier.predict_proba(X)[0])
        alert_label   = "⚠️  HIGH RISK" if alert_proba >= self.classifier.optimal_threshold else "✅  NORMAL"

        return {
            "risk_score":           round(risk_score, 6),
            "collision_probability": float(f"{10**risk_score:.2e}"),
            "high_risk_probability": round(alert_proba, 4),
            "alert":                alert_label,
            "threshold_used":       round(self.classifier.optimal_threshold, 3),
        }

    def predict_batch(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Predict on a CSV of new conjunction events.

        Parameters
        ----------
        csv_path    : path to input CSV
        output_path : if provided, saves results to this path

        Returns
        -------
        DataFrame with original columns + predictions appended
        """
        df_raw = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"Loaded {len(df_raw)} events for batch prediction")

        df = df_raw.copy()
        missing_value_report(df)
        df = self._preprocess(df)

        X_reg = self._align_features(df.copy(), self.regressor.feature_names)
        X_clf = self._align_features(df.copy(), self.classifier.feature_names)

        df_raw["predicted_risk_score"]       = self.regressor.predict(X_reg).round(6)
        df_raw["predicted_collision_prob"]   = (10 ** df_raw["predicted_risk_score"]).apply(
            lambda x: float(f"{x:.2e}")
        )
        df_raw["high_risk_probability"]      = self.classifier.predict_proba(X_clf).round(4)
        df_raw["alert"]                      = (
            df_raw["high_risk_probability"] >= self.classifier.optimal_threshold
        ).map({True: "HIGH_RISK", False: "NORMAL"})

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_raw.to_csv(output_path, index=False)
            logger.info(f"Batch predictions saved → {output_path}")

        high_risk_count = (df_raw["alert"] == "HIGH_RISK").sum()
        logger.info(f"Batch complete: {high_risk_count}/{len(df_raw)} "
                    f"({high_risk_count/len(df_raw)*100:.1f}%) events flagged HIGH RISK")

        return df_raw

    def generate_alert_report(self, df_predictions: pd.DataFrame) -> str:
        """Generate a plain-text alert report from batch predictions."""
        total      = len(df_predictions)
        high_risk  = df_predictions[df_predictions["alert"] == "HIGH_RISK"]
        n_high     = len(high_risk)

        lines = [
            "=" * 60,
            "  SATELLITE CONJUNCTION RISK ALERT REPORT",
            "=" * 60,
            f"  Total events analysed : {total}",
            f"  High-risk alerts      : {n_high} ({n_high/total*100:.1f}%)",
            f"  Normal events         : {total - n_high}",
            "",
        ]

        if n_high > 0:
            lines.append("  TOP HIGH-RISK EVENTS:")
            lines.append("  " + "-" * 50)
            top = high_risk.nlargest(10, "high_risk_probability")
            for _, row in top.iterrows():
                eid = row.get("event_id", "N/A")
                lines.append(
                    f"  Event {eid:<8} | "
                    f"Risk: {row['predicted_risk_score']:.4f} | "
                    f"P(collision): {row['predicted_collision_prob']:.2e} | "
                    f"Alert prob: {row['high_risk_probability']:.3f}"
                )
        else:
            lines.append("  No high-risk events detected.")

        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)
        return report