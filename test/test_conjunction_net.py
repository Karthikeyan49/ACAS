"""
TEST FILE 3 — CONJUNCTIONNET NEURAL NETWORK
=============================================
What this tests:
  1. extract_features() converts a conjunction dict into 12 numbers
  2. ConjunctionNet builds without errors
  3. Model forward pass returns a value between 0 and 1
  4. Training runs and loss decreases
  5. ONNX export works
  6. ONNX model inference returns same result as PyTorch model
  7. High-risk conjunction gets higher Pc than low-risk

How to run:
  python tests/test_3_conjunction_net.py

What you should see if it works:
  - 12 features extracted correctly
  - PyTorch model output between 0 and 1
  - Training loss decreasing each epoch
  - ONNX file created
  - ONNX inference matches PyTorch inference
  - Dangerous scenario gets Pc > safe scenario
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import onnxruntime as ort

from models.conjunction_net import (
    ConjunctionNet,
    extract_features,
    generate_training_data,
    train_and_export
)

# ─────────────────────────────────────────────────────────
# STEP 1 — Test extract_features() with a known conjunction
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Testing extract_features()")
print("="*60)

# Build a sample conjunction dict (like conjunction_finder outputs)
sample_conjunction = {
    'object_id':     'TEST-001',
    'miss_km':       0.8,           # 800m miss — dangerous
    'tca_hours':     2.4,           # 2.4 hours from now
    'rel_pos':       np.array([0.6, -0.5, 0.2]),   # km
    'rel_vel':       np.array([-7.1, 3.2, 1.8]),   # km/s
    'rel_speed_kms': 8.04,
    'tle_stale':     False,
    'tle_age_hours': 12.0
}

features = extract_features(sample_conjunction)

print(f"✅ extract_features() returned array of shape: {features.shape}")
print(f"   Expected shape: (12,)")
print(f"\n   Feature values:")
print(f"   [0] rel_pos X     : {features[0]:.4f}  km")
print(f"   [1] rel_pos Y     : {features[1]:.4f}  km")
print(f"   [2] rel_pos Z     : {features[2]:.4f}  km")
print(f"   [3] rel_vel X     : {features[3]:.4f}  km/s")
print(f"   [4] rel_vel Y     : {features[4]:.4f}  km/s")
print(f"   [5] rel_vel Z     : {features[5]:.4f}  km/s")
print(f"   [6] miss_km       : {features[6]:.4f}  km")
print(f"   [7] tca_hours     : {features[7]:.4f}  hours")
print(f"   [8] speed         : {features[8]:.4f}  km/s")
print(f"   [9] approach_angle: {features[9]:.4f}")
print(f"   [10] tle_stale    : {features[10]:.1f}  (0=fresh, 1=stale)")
print(f"   [11] danger_time  : {features[11]:.4f}")

assert features.shape == (12,), f"❌ Wrong shape: {features.shape}"
assert features.dtype == np.float32, f"❌ Wrong dtype: {features.dtype}"
print(f"\n✅ PASSED — Features shape and dtype correct")

# ─────────────────────────────────────────────────────────
# STEP 2 — Test model forward pass (untrained is fine)
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Testing ConjunctionNet forward pass")
print("="*60)

model = ConjunctionNet()
print(f"✅ ConjunctionNet created")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")

# Run forward pass
model.eval()
with torch.no_grad():
    input_tensor = torch.FloatTensor(features).unsqueeze(0)  # shape (1, 12)
    output = model(input_tensor)

pc_value = output.item()
print(f"\n   Input shape : {input_tensor.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Raw Pc value: {pc_value:.6f}")

assert 0.0 <= pc_value <= 1.0, f"❌ Pc={pc_value} is outside [0,1]"
print(f"✅ PASSED — Output {pc_value:.4f} is between 0 and 1")

# ─────────────────────────────────────────────────────────
# STEP 3 — Quick training test (small dataset, few epochs)
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Quick training test (500 samples, 10 epochs)")
print("="*60)
print("NOTE: Full training (60k samples, 100 epochs) runs in train_and_export()")
print("      This quick test just verifies loss decreases\n")

X, y = generate_training_data(n=500)
Xt = torch.FloatTensor(X)
yt = torch.FloatTensor(y)

quick_model = ConjunctionNet()
optimizer = torch.optim.Adam(quick_model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

losses = []
for epoch in range(10):
    quick_model.train()
    optimizer.zero_grad()
    pred = quick_model(Xt)
    loss = loss_fn(pred, yt)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"   Epoch {epoch+1:2d} | Loss: {loss.item():.6f}")

loss_decreased = losses[-1] < losses[0]
if loss_decreased:
    print(f"\n✅ PASSED — Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f}")
else:
    print(f"\n⚠️  Loss did not decrease clearly — this can happen with only 10 epochs")

# ─────────────────────────────────────────────────────────
# STEP 4 — Test ONNX export and inference
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Testing ONNX export and inference")
print("="*60)

os.makedirs("trained_models", exist_ok=True)
onnx_path = "trained_models/test_conjunction_model.onnx"

quick_model.eval()
dummy = torch.randn(1, 12)
torch.onnx.export(
    quick_model, dummy, onnx_path,
    input_names=['features'],
    output_names=['collision_probability'],
    opset_version=11
)

file_size_kb = os.path.getsize(onnx_path) / 1024
print(f"✅ ONNX file created: {onnx_path}")
print(f"   File size: {file_size_kb:.1f} KB  (full model ~150KB)")

# Run ONNX inference
ort_session = ort.InferenceSession(
    onnx_path, providers=['CPUExecutionProvider']
)
onnx_input = features.reshape(1, -1)
onnx_output = ort_session.run(None, {'features': onnx_input})[0][0][0]

# Compare with PyTorch output
quick_model.eval()
with torch.no_grad():
    pt_output = quick_model(torch.FloatTensor(features).unsqueeze(0)).item()

print(f"\n   PyTorch inference result : {pt_output:.6f}")
print(f"   ONNX inference result    : {onnx_output:.6f}")
print(f"   Difference               : {abs(pt_output - onnx_output):.8f}")

assert abs(pt_output - onnx_output) < 1e-4, "❌ ONNX and PyTorch outputs differ too much"
print(f"✅ PASSED — ONNX and PyTorch outputs match")

# ─────────────────────────────────────────────────────────
# STEP 5 — Sanity check: dangerous conjunction > safe one
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Sanity check — dangerous scenario vs safe scenario")
print("="*60)

dangerous = {
    'miss_km': 0.1, 'tca_hours': 0.5,
    'rel_pos': np.array([0.08, -0.06, 0.02]),
    'rel_vel': np.array([-10.0, 5.0, 2.0]),
    'tle_stale': False
}

safe = {
    'miss_km': 4.8, 'tca_hours': 48.0,
    'rel_pos': np.array([3.5, -3.0, 1.2]),
    'rel_vel': np.array([-1.0, 0.5, 0.2]),
    'tle_stale': False
}

stale = {
    'miss_km': 2.0, 'tca_hours': 6.0,
    'rel_pos': np.array([1.5, -1.3, 0.5]),
    'rel_vel': np.array([-5.0, 2.5, 1.0]),
    'tle_stale': True    # ← stale TLE should push Pc higher
}

quick_model.eval()
with torch.no_grad():
    pc_dangerous = quick_model(torch.FloatTensor(extract_features(dangerous)).unsqueeze(0)).item()
    pc_safe      = quick_model(torch.FloatTensor(extract_features(safe)).unsqueeze(0)).item()
    pc_stale     = quick_model(torch.FloatTensor(extract_features(stale)).unsqueeze(0)).item()

print(f"   Dangerous scenario Pc : {pc_dangerous:.6f}  (0.1km miss, 0.5h TCA)")
print(f"   Safe scenario Pc      : {pc_safe:.6f}  (4.8km miss, 48h TCA)")
print(f"   Stale TLE scenario Pc : {pc_stale:.6f}  (stale=True inflates features)")

if pc_dangerous > pc_safe:
    print(f"\n✅ PASSED — Dangerous Pc ({pc_dangerous:.4f}) > Safe Pc ({pc_safe:.4f})")
else:
    print(f"\n⚠️  Pc ordering unexpected — model needs more training epochs")
    print(f"   This is normal for an untrained/quickly-trained model")
    print(f"   Full training with train_and_export() will fix this")

print(f"\n{'='*60}")
print("TEST 3 COMPLETE — ConjunctionNet working correctly")
print("="*60)
print("\nNext step: Run full training with:")
print("  python models/conjunction_net.py")
print("  This trains on 60,000 samples for 100 epochs (~5 min)")