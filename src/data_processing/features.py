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

def compute_roi_metrics(traj):
    traj = np.array(traj, dtype=np.float32)  # shape (T, 2)
    diffs = np.diff(traj, axis=0)
    velocity = np.linalg.norm(diffs, axis=1)
    acceleration = np.diff(velocity)
    jerk = np.diff(acceleration)

    covered_distance = velocity.sum()
    displacement = np.linalg.norm(traj[-1] - traj[0])
    extent = np.ptp(traj, axis=0)  # peak-to-peak (max - min) per axis

    return [
        covered_distance,
        displacement,
        *extent,
        jerk.mean() if jerk.size else 0.0,
        jerk.std() if jerk.size else 0.0
    ]

def extract_features_from_window(window, window_idx=None):
    angles, velocities, acceleration = [], [], []
    symmetry, spreads, motion_energy, joint_stddevs = [], [], [], []
    roi_trajectories = {"head": [], "lhand": [], "rhand": [], "trunk": []}

    prev_frame = None
    prev_velocity = None

    for frame_idx, frame in enumerate(window):
        frame = np.array(frame, dtype=np.float32)
        if frame.ndim != 2 or frame.shape[0] != 17 or frame.shape[1] < 2:
            raise ValueError(f"Malformed frame at window {window_idx}, frame {frame_idx}")

        frame_xy = frame[:, :2]

        # Joint angles
        angles.append([
            angle_between(frame[5], frame[7], frame[9]),
            angle_between(frame[6], frame[8], frame[10]),
            angle_between(frame[11], frame[13], frame[15]),
            angle_between(frame[12], frame[14], frame[16])
        ])

        # ROI tracking
        roi_trajectories["head"].append(frame[0])
        roi_trajectories["lhand"].append(frame[9])
        roi_trajectories["rhand"].append(frame[10])
        roi_trajectories["trunk"].append(np.mean(frame[[5, 6, 11, 12]], axis=0))

        # Velocity
        if prev_frame is not None:
            velocity = np.linalg.norm(frame[[9, 10, 15, 16]] - prev_frame[[9, 10, 15, 16]], axis=1)
            velocities.append(velocity)

            if prev_velocity is not None:
                accel = np.abs(velocity - prev_velocity)
                acceleration.append(accel)
            prev_velocity = velocity
        else:
            velocities.append(np.zeros(4, dtype=np.float32))

        # Symmetry
        symmetry.append([
            np.linalg.norm(frame[9] - frame[10]),
            np.linalg.norm(frame[15] - frame[16])
        ])

        spreads.append([np.linalg.norm(frame[9] - frame[10])])

        if prev_frame is not None:
            motion_energy.append(np.linalg.norm(frame_xy - prev_frame[:, :2], axis=1).sum())
        else:
            motion_energy.append(0.0)

        joint_stddevs.append(np.std(frame_xy))
        prev_frame = frame

    # Convert to arrays
    angles = np.array(angles, dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    acceleration = np.array(acceleration, dtype=np.float32) if acceleration else np.zeros_like(velocities)
    symmetry = np.array(symmetry, dtype=np.float32)
    spreads = np.array(spreads, dtype=np.float32)
    motion_energy = np.array(motion_energy, dtype=np.float32)
    joint_stddevs = np.array(joint_stddevs, dtype=np.float32)

    # ROI features
    roi_features = []
    for key in ["head", "lhand", "rhand", "trunk"]:
        roi_features.extend(compute_roi_metrics(roi_trajectories[key]))

    return np.concatenate([
        angles.mean(axis=0), angles.std(axis=0),
        velocities.mean(axis=0), velocities.std(axis=0),
        acceleration.mean(axis=0), acceleration.std(axis=0),
        symmetry.mean(axis=0), symmetry.std(axis=0),
        spreads.mean(axis=0), spreads.std(axis=0),
        [motion_energy.mean(), motion_energy.std()],
        [joint_stddevs.mean(), joint_stddevs.std()],
        roi_features
    ])

def main():
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
    data_path = os.path.join(PROJECT_ROOT, "data", "processed_windows")

    X_raw_path = os.path.join(data_path, "X_windows.pkl")
    y_path = os.path.join(data_path, "y_labels.pkl")
    out_path = os.path.join(data_path, "X_features.pkl")

    print("Loading preprocessed pose windows...")
    X_raw = joblib.load(X_raw_path)
    y = joblib.load(y_path)

    print(f"Processing {len(X_raw)} windows…")
    X_features = []
    bad_count = 0

    for w_idx, window in enumerate(tqdm(X_raw)):
        try:
            feats = extract_features_from_window(window, window_idx=w_idx)
            X_features.append(feats)
        except ValueError as ve:
            bad_count += 1
            print(f"[Warning] Skipping window {w_idx} because: {ve}")

    X_features = np.array(X_features, dtype=np.float32)
    joblib.dump(X_features, out_path)

    print(f"Saved extracted features to {out_path}")
    print(f"Feature matrix shape: {X_features.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    print(f"Skipped {bad_count} windows due to malformed frames.")

if __name__ == "__main__":
    main()
