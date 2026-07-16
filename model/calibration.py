"""
model/calibration.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Probability-calibration infrastructure for the LightGBM collision
    probability. A raw model score is not necessarily a well-calibrated
    probability — of all conjunctions the model calls "1e-4", the true fraction
    that would collide may be higher or lower. This file fits an isotonic
    regression that remaps raw_pc → calibrated_pc using historical outcomes,
    and applies it at runtime when a trained calibrator is present.

    Ground software (offline fitting + plotting), so sklearn / matplotlib are
    fair game here — unlike core/, which must stay numpy + stdlib only.

CALLED FROM
    model/lgbm_engine.py    calibration.apply_if_available(pc)  — final Pc step
    CLI:  python model/calibration.py --fit data.csv

CALLS INTO
    numpy, pickle
    sklearn.isotonic.IsotonicRegression   (fit / transform)
    matplotlib                            (reliability diagram, optional)

WHAT IT PROVIDES
    PcCalibrator
        fit(raw_pc_array, actual_labels_or_pc)   monotone isotonic remap
        transform(raw_pc)                        scalar or array → calibrated
        save(path) / load(path)                  pickle round-trip
        is_fitted -> bool

    apply_if_available(raw_pc) -> float
        Loads trained_models/lgbm/calibrator.pkl once (cached). If absent or
        unloadable, returns raw_pc unchanged (identity) — the flight-safe path
        when no calibration data ships with the repo.

    fit_from_csv(csv_path) -> PcCalibrator
        Reads a CSV with columns (raw_pc, actual) [flexible names], fits a
        calibrator, saves it next to the model, and writes a reliability
        diagram PNG to outputs/plots/.

CSV COLUMNS (fit_from_csv, case-insensitive, first match wins)
    raw_pc   : raw_pc | pc | pred | prediction | raw
    actual   : actual | label | y | outcome | collision | true_pc
══════════════════════════════════════════════════════════════════════════════
"""
import os
import pickle

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

DEFAULT_CALIBRATOR_PATH = os.path.join(
    _PROJECT_ROOT, "trained_models", "lgbm", "calibrator.pkl"
)
DEFAULT_PLOT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "plots")

# Isotonic operates on probabilities; clamp inputs into a safe open interval so
# log-scaled reliability plots and boundary values stay finite.
_EPS = 1e-12


class PcCalibrator:
    """Isotonic-regression calibrator mapping raw Pc → calibrated Pc."""

    def __init__(self):
        self._iso = None          # sklearn IsotonicRegression once fitted
        self.n_samples = 0

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(self, raw_pc_array, actual_labels_or_pc):
        """Fit the monotone remap.

        raw_pc_array         : model outputs in [0, 1].
        actual_labels_or_pc  : observed outcomes — either binary {0,1} labels
                               or empirical probabilities in [0, 1].
        """
        from sklearn.isotonic import IsotonicRegression

        x = np.clip(np.asarray(raw_pc_array, dtype=float).ravel(), _EPS, 1.0)
        y = np.clip(np.asarray(actual_labels_or_pc, dtype=float).ravel(), 0.0, 1.0)
        if x.shape != y.shape or x.size == 0:
            raise ValueError("raw_pc_array and actual must be non-empty, same length")

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(x, y)
        self._iso = iso
        self.n_samples = int(x.size)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None

    # ── applying ─────────────────────────────────────────────────────────────
    def transform(self, raw_pc):
        """Remap raw Pc → calibrated Pc. Identity until fitted.

        Accepts a scalar or array; returns the same shape (scalar in → float).
        """
        if self._iso is None:
            return raw_pc
        scalar = np.isscalar(raw_pc)
        x = np.clip(np.asarray(raw_pc, dtype=float).ravel(), _EPS, 1.0)
        y = np.clip(self._iso.predict(x), 0.0, 1.0)
        return float(y[0]) if scalar else y.reshape(np.asarray(raw_pc).shape)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str = None):
        path = path or DEFAULT_CALIBRATOR_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str = None) -> "PcCalibrator":
        path = path or DEFAULT_CALIBRATOR_PATH
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} did not contain a PcCalibrator")
        return obj


# ── runtime helper ───────────────────────────────────────────────────────────
_RUNTIME_CACHE = {"loaded": False, "calibrator": None}


def apply_if_available(raw_pc, path: str = None):
    """Apply the trained calibrator if one exists, else return raw_pc unchanged.

    The load is attempted once and cached (including the "absent" result), so
    the hot inference path pays no repeated filesystem cost.
    """
    path = path or DEFAULT_CALIBRATOR_PATH
    if not _RUNTIME_CACHE["loaded"]:
        _RUNTIME_CACHE["loaded"] = True
        try:
            if os.path.exists(path):
                _RUNTIME_CACHE["calibrator"] = PcCalibrator.load(path)
        except Exception:
            _RUNTIME_CACHE["calibrator"] = None

    cal = _RUNTIME_CACHE["calibrator"]
    if cal is None or not cal.is_fitted:
        return raw_pc
    return cal.transform(raw_pc)


def _reset_runtime_cache():
    """Test hook — force apply_if_available to re-check the filesystem."""
    _RUNTIME_CACHE["loaded"] = False
    _RUNTIME_CACHE["calibrator"] = None


# ── CLI: fit from a CSV of (raw_pc, actual) ──────────────────────────────────
_RAW_ALIASES = ("raw_pc", "pc", "pred", "prediction", "raw")
_ACTUAL_ALIASES = ("actual", "label", "y", "outcome", "collision", "true_pc")


def _pick_column(columns, aliases):
    lower = {c.lower(): c for c in columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _reliability_diagram(raw, actual, calibrated, out_path, n_bins=10):
    """Write a reliability diagram PNG. Best-effort — skipped if matplotlib
    is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    def _binned(pred, obs):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = idx == b
            if np.any(m):
                xs.append(pred[m].mean())
                ys.append(obs[m].mean())
        return np.array(xs), np.array(ys)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    rx, ry = _binned(raw, actual)
    cx, cy = _binned(calibrated, actual)
    ax.plot(rx, ry, "o-", label="raw", color="#d1495b")
    ax.plot(cx, cy, "s-", label="calibrated", color="#2e7d32")
    ax.set_xlabel("Predicted Pc")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram — LightGBM Pc calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fit_from_csv(csv_path: str, out_path: str = None, plot_dir: str = None) -> PcCalibrator:
    """Fit a calibrator from a CSV and persist it plus a reliability diagram."""
    import csv

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        raw_col = _pick_column(cols, _RAW_ALIASES)
        act_col = _pick_column(cols, _ACTUAL_ALIASES)
        if raw_col is None or act_col is None:
            raise ValueError(
                f"CSV must have a raw-Pc column {_RAW_ALIASES} and an actual "
                f"column {_ACTUAL_ALIASES}; found {cols}"
            )
        raw, actual = [], []
        for row in reader:
            try:
                raw.append(float(row[raw_col]))
                actual.append(float(row[act_col]))
            except (TypeError, ValueError):
                continue

    raw = np.asarray(raw)
    actual = np.asarray(actual)
    cal = PcCalibrator().fit(raw, actual)

    out_path = out_path or DEFAULT_CALIBRATOR_PATH
    cal.save(out_path)

    plot_dir = plot_dir or DEFAULT_PLOT_DIR
    png = os.path.join(plot_dir, "pc_reliability_diagram.png")
    _reliability_diagram(raw, actual, cal.transform(raw), png)

    return cal


def _main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Fit the LightGBM Pc calibrator.")
    p.add_argument("--fit", metavar="CSV", help="CSV of (raw_pc, actual) to fit from")
    p.add_argument("--out", default=None, help="calibrator output path (.pkl)")
    args = p.parse_args(argv)

    if args.fit:
        cal = fit_from_csv(args.fit, out_path=args.out)
        print(f"Fitted calibrator on {cal.n_samples} samples → "
              f"{args.out or DEFAULT_CALIBRATOR_PATH}")
        print(f"Reliability diagram → "
              f"{os.path.join(DEFAULT_PLOT_DIR, 'pc_reliability_diagram.png')}")
    else:
        p.print_help()


if __name__ == "__main__":
    _main()
