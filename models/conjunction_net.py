# ─────────────────────────────────────────────────────────────────────────────
# models/conjunction_net.py
#
# PURPOSE:
#   Neural network that predicts Probability of Collision (Pc)
#   from 12 features of a conjunction event.
#
#   Also contains:
#     extract_features()       — converts a conjunction dict → 12 numbers
#     generate_training_data() — creates synthetic training dataset
#     train_and_export()       — trains the model and saves as ONNX
#
# CALLED BY:
#   onboard/acas_controller.py — loads the ONNX file for inference
#
# TO TRAIN:
#   python models/conjunction_net.py
#   Saves: trained_models/conjunction_model.onnx
#          trained_models/conjunction_model.pt
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# ConjunctionNet
# 4-layer neural network: 12 inputs → 1 output (Pc between 0 and 1)
# ─────────────────────────────────────────────────────────────────────────────
class ConjunctionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 12 → 64
            # BatchNorm stabilises training when features have different scales
            # ReLU introduces non-linearity (can learn curved relationships)
            # Dropout randomly disables 30% of neurons to prevent overfitting
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 64 → 128  (expand to capture more patterns)
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3: 128 → 64  (compress back down)
            nn.Linear(128, 64),
            nn.ReLU(),

            # Layer 4: 64 → 1  (single Pc output)
            # Sigmoid squashes any number into [0, 1] range
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# extract_features
# Converts one conjunction event dictionary into a 12-element numpy array
# that the neural network can consume.
#
# INPUT:  conjunction dict from ConjunctionFinder.find_all()
# OUTPUT: np.array of shape (12,) dtype float32
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(c: dict) -> np.ndarray:
    rp  = c['rel_pos']                         # relative position (km)
    rv  = c['rel_vel']                         # relative velocity (km/s)
    md  = c['miss_km']                         # miss distance (km)
    tca = c['tca_hours']                       # hours to TCA
    spd = np.linalg.norm(rv)                   # closing speed (km/s)

    # Approach angle: -1 = head-on, 0 = perpendicular, +1 = tail-chase
    rp_norm = np.linalg.norm(rp)
    angle   = np.dot(rp, rv) / (rp_norm * spd + 1e-10)

    # How long the satellite spends in the danger zone (hours)
    danger_time = md / (spd + 1e-10)

    return np.array([
        rp[0], rp[1], rp[2],          # features 0-2:  relative position
        rv[0], rv[1], rv[2],          # features 3-5:  relative velocity
        md,                            # feature  6:    miss distance
        tca,                           # feature  7:    time to TCA
        spd,                           # feature  8:    closing speed
        angle,                         # feature  9:    approach geometry
        float(c.get('tle_stale', 0)), # feature  10:   staleness flag (LIMITATION)
        danger_time                    # feature  11:   time in danger zone
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# generate_training_data
# Creates synthetic conjunction scenarios with physics-based Pc labels.
# We use synthetic data because there is no large public dataset of
# real conjunction events with known collision outcomes.
# ─────────────────────────────────────────────────────────────────────────────
def generate_training_data(n: int = 60000):
    X, y = [], []

    for _ in range(n):
        # Randomise scenario parameters
        miss  = np.random.exponential(2.5)          # km — most are moderate
        tca   = np.random.uniform(0.3, 72)          # hours
        speed = np.random.uniform(0.5, 15)          # km/s (LEO range)
        stale = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% stale

        rp = np.random.randn(3)
        rp = rp / np.linalg.norm(rp) * miss         # direction × magnitude

        rv = np.random.randn(3)
        rv = rv / np.linalg.norm(rv) * speed

        # Physics-based Pc label
        # Smaller miss + faster approach = higher probability
        obj_size_km = 0.01   # 10 metre object
        pc = min((obj_size_km / (miss + 1e-10))**2 * speed / 7.8, 1.0)

        # Stale TLE = uncertainty → treat as more dangerous
        if stale:
            pc = min(pc * 2.5, 1.0)

        c = {
            'rel_pos': rp, 'rel_vel': rv,
            'miss_km': miss, 'tca_hours': tca,
            'tle_stale': bool(stale)
        }
        X.append(extract_features(c))
        y.append([pc])

    return np.array(X), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# train_and_export
# Full training pipeline:
#   1. Generate 60,000 synthetic conjunction scenarios
#   2. Train ConjunctionNet for 100 epochs
#   3. Export to ONNX (for onboard inference on satellite OBC)
#   4. Save PyTorch checkpoint (for retraining)
#
# Runtime: ~5 minutes on CPU
# ─────────────────────────────────────────────────────────────────────────────
def train_and_export():
    os.makedirs("trained_models", exist_ok=True)

    print("⏳ Generating 60,000 training scenarios...")
    X, y      = generate_training_data(60000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model   = ConjunctionNet()
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched   = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    loss_fn = nn.MSELoss()

    Xt = torch.FloatTensor(Xtr)
    yt = torch.FloatTensor(ytr)

    print("⏳ Training ConjunctionNet (100 epochs)...")
    for epoch in range(100):
        model.train()
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(
                    model(torch.FloatTensor(Xte)),
                    torch.FloatTensor(yte)
                ).item()
            print(f"  Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | "
                  f"Test Loss: {test_loss:.6f}")

    # Export to ONNX — this is the format that runs on the satellite OBC
    # ONNX Runtime is lightweight (~5MB) and runs on ARM processors
    # Inference time: < 5ms per conjunction on ARM Cortex-A53
    model.eval()
    dummy_input = torch.randn(1, 12)
    torch.onnx.export(
        model,
        dummy_input,
        "trained_models/conjunction_model.onnx",
        input_names=['features'],
        output_names=['collision_probability'],
        opset_version=18
    )

    # Save PyTorch weights for future retraining
    torch.save(model.state_dict(), "trained_models/conjunction_model.pt")

    onnx_kb = os.path.getsize("trained_models/conjunction_model.onnx") / 1024
    print(f"✅ ConjunctionNet trained and saved")
    print(f"   ONNX file: trained_models/conjunction_model.onnx ({onnx_kb:.1f} KB)")
    print(f"   PT file  : trained_models/conjunction_model.pt")
    return model


if __name__ == "__main__":
    train_and_export()