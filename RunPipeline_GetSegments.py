import os
import numpy as np
import joblib
import cv2
from src.pose_extraction.movenet_extractor import MoveNetPoseExtractor
from src.data_processing.features import extract_features_from_window  # note: new signature

def extract_windows_from_video(video_path, window_size=60, stride=30):
    extractor     = MoveNetPoseExtractor()
    landmarks_seq = extractor.extract_landmarks(video_path)
    
    # Discard frames that are None
    landmarks_seq = [l for l in landmarks_seq if l is not None]
    if len(landmarks_seq) < window_size:
        return [], []

    windows       = []
    frame_indices = []
    for i in range(0, len(landmarks_seq) - window_size + 1, stride):
        window = np.stack(landmarks_seq[i:i + window_size])
        windows.append(window)
        frame_indices.append(i)

    return windows, frame_indices

def predict_seizure_ranges(video_path, model_dir):
    # Load KNN model and scaler
    model  = joblib.load(os.path.join(model_dir, "knn.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))

    # Extract windows from video
    windows, start_frames = extract_windows_from_video(video_path)
    if not windows:
        print("Not enough frames in video to extract windows.")
        return []

    # Extract features for each window, skipping bad ones
    X_features = []
    valid_start_frames = []
    for w_idx, window in enumerate(windows):
        try:
            feats = extract_features_from_window(window, window_idx=w_idx)
            X_features.append(feats)
            valid_start_frames.append(start_frames[w_idx])
        except ValueError as ve:
            print(f"[Predict] Skipping window {w_idx} due to malformed frame: {ve}")
            continue

    if len(X_features) == 0:
        print("All windows were malformed or too small; no features to predict.")
        return []

    X_features = np.array(X_features, dtype=np.float32)
    X_scaled   = scaler.transform(X_features)

    # Predict
    y_pred = model.predict(X_scaled)

    # Convert predicted windows to time segments
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    seizure_segments = []
    current_segment = None

    for idx, label in enumerate(y_pred):
        start_frame = valid_start_frames[idx]
        end_frame   = start_frame + 60
        start_time  = start_frame / fps
        end_time    = end_frame   / fps

        if label == 1:
            if current_segment is None:
                current_segment = [start_time, end_time]
            else:
                current_segment[1] = end_time
        else:
            if current_segment is not None:
                seizure_segments.append(tuple(current_segment))
                current_segment = None

    # Append last open segment
    if current_segment is not None:
        seizure_segments.append(tuple(current_segment))

    return seizure_segments

# Example usage
if __name__ == "__main__":
    video_path = "path/to/video.mp4"  # Replace when running
    model_path = "models"
    results    = predict_seizure_ranges(video_path, model_path)

    for i, (start, end) in enumerate(results):
        print(f"Seizure {i+1}: Start = {start:.2f}s, End = {end:.2f}s")
