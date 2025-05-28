import os
import joblib
import numpy as np
from tqdm import tqdm
from math import acos, degrees

def angle_between(p1, p2, p3):
    a = np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
    b = np.array(p3, dtype=np.float32) - np.array(p2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = np.dot(a, b) / (norm_a * norm_b)
    cosine = np.clip(cosine, -1.0, 1.0)
    return degrees(acos(cosine))

def extract_features_from_window(window, window_idx=None):
    """
    window: a list/array of shape (n_frames, 17, ≥2)
    window_idx: optional integer, for debug printing
    """
    angles = []
    velocities = []
    symmetry = []
    spreads = []
    motion_energy = []
    joint_stddevs = []

    prev_frame = None
    prev_velocity = None
    acceleration = []

    for frame_idx, frame in enumerate(window):
        # Force it to float32 NumPy array
        frame = np.array(frame, dtype=np.float32)

        # Debug: check shape now
        expected_msg = f"(17, ≥2)"
        actual_shape = frame.shape
        if frame.ndim != 2 or frame.shape[0] != 17 or frame.shape[1] < 2:
            # Print a clear debug line, including which window and frame
            if window_idx is None:
                print(f"[Debug] frame_idx={frame_idx}, got shape {actual_shape}, expected {expected_msg}")
            else:
                print(f"[Debug] window {window_idx}, frame {frame_idx}, got shape {actual_shape}, expected {expected_msg}")
            raise ValueError(f"Malformed frame at window {window_idx}, index {frame_idx}: got shape {actual_shape}, expected {expected_msg}")

        frame_xy = frame[:, :2]  # we only need x,y for angles, etc.

        # --- 1. Joint angles ---
        angles.append([
            angle_between(frame[5][:2],  frame[7][:2],  frame[9][:2]),    # L elbow
            angle_between(frame[6][:2],  frame[8][:2],  frame[10][:2]),   # R elbow
            angle_between(frame[11][:2], frame[13][:2], frame[15][:2]),   # L knee
            angle_between(frame[12][:2], frame[14][:2], frame[16][:2])    # R knee
        ])

        # --- 2. Velocity (if previous frame available) ---
        if prev_frame is not None:
            v_lwrist  = np.linalg.norm(frame[9][:2]  - prev_frame[9][:2])
            v_rwrist  = np.linalg.norm(frame[10][:2] - prev_frame[10][:2])
            v_lankle  = np.linalg.norm(frame[15][:2] - prev_frame[15][:2])
            v_rankle  = np.linalg.norm(frame[16][:2] - prev_frame[16][:2])
            velocity  = np.array([v_lwrist, v_rwrist, v_lankle, v_rankle], dtype=np.float32)
            velocities.append(velocity)

            # --- 3. Acceleration
            if prev_velocity is not None:
                accel = np.abs(velocity - prev_velocity)
                acceleration.append(accel)
            prev_velocity = velocity
        else:
            velocities.append(np.zeros(4, dtype=np.float32))

        # --- 4. Symmetry: distance between left/right wrists & ankles
        sym_wrist = np.linalg.norm(frame[9][:2]  - frame[10][:2])
        sym_ankle = np.linalg.norm(frame[15][:2] - frame[16][:2])
        symmetry.append([sym_wrist, sym_ankle])

        # --- 5. Spread (wrist-to-wrist)
        wrist_spread = np.linalg.norm(frame[9][:2] - frame[10][:2])
        spreads.append([wrist_spread])

        # --- 6. Motion energy: total keypoint movement from prev frame
        if prev_frame is not None:
            total_movement = np.linalg.norm(frame_xy - prev_frame[:, :2], axis=1).sum()
            motion_energy.append(total_movement)
        else:
            motion_energy.append(0.0)

        # --- 7. Joint position variation (pose instability)
        joint_stddevs.append(np.std(frame_xy))

        prev_frame = frame

    # Convert lists → arrays (all dtype=float32 now)
    angles        = np.array(angles,        dtype=np.float32)
    velocities    = np.array(velocities,    dtype=np.float32)
    acceleration  = np.array(acceleration,  dtype=np.float32) if acceleration else np.zeros_like(velocities)
    symmetry      = np.array(symmetry,      dtype=np.float32)
    spreads       = np.array(spreads,       dtype=np.float32)
    motion_energy = np.array(motion_energy, dtype=np.float32)
    joint_stddevs = np.array(joint_stddevs, dtype=np.float32)

    # Aggregate: mean + std for each feature group
    features = np.concatenate([
        angles.mean(axis=0),        angles.std(axis=0),
        velocities.mean(axis=0),    velocities.std(axis=0),
        acceleration.mean(axis=0),  acceleration.std(axis=0),
        symmetry.mean(axis=0),      symmetry.std(axis=0),
        spreads.mean(axis=0),       spreads.std(axis=0),
        [motion_energy.mean(), motion_energy.std()],
        [joint_stddevs.mean(), joint_stddevs.std()]
    ], axis=0)

    return features

def main():
    SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
    data_path    = os.path.join(PROJECT_ROOT, "data", "processed_windows")

    X_raw_path = os.path.join(data_path, "X_windows.pkl")
    y_path     = os.path.join(data_path, "y_labels.pkl")
    out_path   = os.path.join(data_path, "X_features.pkl")

    print("Loading preprocessed pose windows...")
    X_raw = joblib.load(X_raw_path)
    y     = joblib.load(y_path)

    print(f"Processing {len(X_raw)} windows…")
    X_features = []
    bad_count  = 0

    for w_idx, window in enumerate(tqdm(X_raw)):
        try:
            feats = extract_features_from_window(window, window_idx=w_idx)
            X_features.append(feats)
        except ValueError as ve:
            bad_count += 1
            print(f"  [Warning] Skipping window {w_idx} because: {ve}")
            continue

    X_features = np.array(X_features, dtype=np.float32)
    joblib.dump(X_features, out_path)

    print(f" Saved extracted features to {out_path}")
    print(f" Feature matrix shape: {X_features.shape}")
    print(f" Label distribution: {np.bincount(y)}")
    print(f" Skipped {bad_count} windows due to malformed frames.")
