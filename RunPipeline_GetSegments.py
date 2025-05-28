import os
import numpy as np
import joblib
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import medfilt

from src.pose_extraction.movenet_extractor import MoveNetPoseExtractor
from src.data_processing.features import extract_features_from_window

def extract_windows_from_video(video_path, window_size=60, stride=30):
    extractor     = MoveNetPoseExtractor()
    landmarks_seq = extractor.extract_landmarks(video_path)
    
    landmarks_seq = [l for l in landmarks_seq if l is not None]
    
    if len(landmarks_seq) < window_size:
        print(f"Warning: Not enough valid frames ({len(landmarks_seq)}) to extract windows of size {window_size}.")
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
    model_path = os.path.join(model_dir, "best_model_knn.pkl")
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")

    if not os.path.exists(model_path):
        print(f"Error: KNN model not found at {model_path}")
        return []
    if not os.path.exists(scaler_path):
        print(f"Error: Feature scaler not found at {scaler_path}")
        return []

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Extract windows from video
    windows, start_frames = extract_windows_from_video(video_path)
    if not windows:
        print("Not enough frames in video to extract windows for prediction.")
        return []

    # Extract features for each window, skipping bad ones
    X_features = []
    valid_start_frames = []
    for w_idx, window in enumerate(windows):
        try:
            # Ensure the window is a valid numpy array before passing to feature extraction
            if not isinstance(window, np.ndarray) or window.size == 0:
                raise ValueError("Window is empty or not a numpy array.")
            feats = extract_features_from_window(window, window_idx=w_idx)
            X_features.append(feats)
            valid_start_frames.append(start_frames[w_idx])
        except ValueError as ve:
            print(f"[Predict] Skipping window {w_idx} due to malformed frame or feature extraction error: {ve}")
            continue

    if len(X_features) == 0:
        print("All windows were malformed or too small; no features to predict.")
        return []

    X_features = np.array(X_features, dtype=np.float32)
    X_scaled   = scaler.transform(X_features)

    # Predict
    # Apply temporal smoothing
    y_pred_raw = model.predict(X_scaled)
    if len(y_pred_raw) >= 5:
        y_pred = medfilt(y_pred_raw, kernel_size=5)
    else:
        y_pred = y_pred_raw


    # Convert predicted windows to time segments
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path} for FPS calculation.")
        return []
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

    if current_segment is not None:
        seizure_segments.append(tuple(current_segment))

    return seizure_segments

def process_video_with_keypoints(input_video_path, output_video_path, seizure_segments=None):

    if seizure_segments is None:
        seizure_segments = []

    extractor = MoveNetPoseExtractor()
    cap       = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_video_path}")

    original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps             = cap.get(cv2.CAP_PROP_FPS)

    output_video_width, output_video_height = original_width, original_height

    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out    = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_width, output_video_height))

    if not out.isOpened():
        print("Warning: 'avc1' codec not available, trying 'mp4v'.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_width, output_video_height))
        if not out.isOpened():
            raise IOError(f"Cannot open video writer for {output_video_path} with 'avc1' or 'mp4v' codec.")


    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head (nose, eyes, ears)
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Torso and arms (shoulders, elbows, wrists)
        (5, 11), (6, 12), # Shoulders to hips
        (11, 13), (13, 15), (12, 14), (14, 16) # Legs (hips, knees, ankles)
    ]

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_count / fps

        keypoints = extractor._process_frame(frame) 

        current_frame_height, current_frame_width, _ = frame.shape 

        # Draw keypoints and connections on the frame
        if keypoints is not None and keypoints.shape == (17, 3):
            # Draw keypoints
            for i in range(keypoints.shape[0]):
                y_norm, x_norm, conf = keypoints[i][0], keypoints[i][1], keypoints[i][2]
                
                if conf > 0.3: # Only draw if confidence is high enough
                    center_x = int(x_norm * current_frame_width)
                    center_y = int(y_norm * current_frame_height)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

            # Draw connections (skeleton)
            for start_node, end_node in connections:
                p1 = keypoints[start_node]
                p2 = keypoints[end_node]
                
                if p1[2] > 0.3 and p2[2] > 0.3:
                    x1_norm, y1_norm = p1[1], p1[0]
                    x2_norm, y2_norm = p2[1], p2[0]

                    cv2.line(frame,
                             (int(x1_norm * current_frame_width), int(y1_norm * current_frame_height)),
                             (int(x2_norm * current_frame_width), int(y2_norm * current_frame_height)),
                             (255, 0, 0), 2)

        display_seizure_message = False
        for start_s, end_s in seizure_segments:
            if start_s <= current_time_sec <= end_s:
                display_seizure_message = True
                break

        if display_seizure_message:
            text = "Possible seizure detected!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_color = (0, 0, 255)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = 10
            text_y = text_size[1] + 10 
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_video_path}")
