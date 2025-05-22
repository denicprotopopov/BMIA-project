import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import joblib
from datetime import datetime, timedelta

def parse_time(t):
    if pd.isnull(t):
        return None
    if isinstance(t, str):
        t = datetime.strptime(t.strip(), '%H:%M:%S').time()
    return t  # already a datetime.time or similar

def load_metadata(excel_path):
    df = pd.read_excel(excel_path, sheet_name="main")
    df.columns = df.columns.str.strip()
    video_info = {}

    for _, row in df.iterrows():
        video_name = row["Name"].replace(".mp4", "_movenet.csv")
        start = parse_time(row["Start"])
        end = parse_time(row["End"])
        duration = parse_time(row["Duration"])
        fps = row["FPS"]

        if pd.isnull(start) or pd.isnull(fps):
            continue

        # If end is missing, assume seizure goes until end of video
        if pd.isnull(end):
            end = duration

        seizure_start = timedelta(hours=start.hour, minutes=start.minute, seconds=start.second).total_seconds()
        seizure_end = timedelta(hours=end.hour, minutes=end.minute, seconds=end.second).total_seconds()

        video_info[video_name] = {
            "seizure_start": seizure_start,
            "seizure_end": seizure_end,
            "fps": fps
        }

    return video_info



def extract_labeled_windows(csv_path, seizure_info, window_size=60, stride=30):
    df = pd.read_csv(csv_path)
    if df.empty or df.shape[1] != 51:
        print(f"Skipped empty or malformed file: {csv_path}")
        return [], []

    data = df.to_numpy().reshape(-1, 17, 3)
    fps = seizure_info["fps"]
    seizure_start = seizure_info["seizure_start"]
    seizure_end = seizure_info["seizure_end"]

    n_frames = len(data)
    segments, labels = [], []

    for i in range(0, n_frames - window_size + 1, stride):
        segment = data[i:i+window_size]
        time_start = i / fps
        time_end = (i + window_size) / fps

        label = int(time_end > seizure_start and time_start < seizure_end)
        segments.append(segment)
        labels.append(label)

    return segments, labels

def build_dataset(landmark_dir, excel_path):
    video_info = load_metadata(excel_path)
    X, y = [], []

    for video_file in tqdm(os.listdir(landmark_dir)):
        if not video_file.endswith(".csv"):
            continue
        if video_file not in video_info:
            continue

        try:
            path = os.path.join(landmark_dir, video_file)
            segments, labels = extract_labeled_windows(path, video_info[video_file])
            X.extend(segments)
            y.extend(labels)
        except Exception as e:
            print(f"Failed to process {video_file}: {e}")

    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Set paths
    landmark_dir = os.path.join("data", "seizure_keypoints")
    excel_path = os.path.join("data", "seizure_metadata.xlsx")

    # Run preprocessing
    X_raw, y = build_dataset(landmark_dir, excel_path)

    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
    output_path = os.path.join(PROJECT_ROOT, "data", "processed_windows")
    os.makedirs(output_path, exist_ok=True)

    # Save preprocessed data
    joblib.dump(X_raw, os.path.join(output_path, "X_windows.pkl"))
    joblib.dump(y, os.path.join(output_path, "y_labels.pkl"))

    print(f"Saved {len(X_raw)} windows to {os.path.abspath(output_path)}")
    print(f"Seizure windows: {np.sum(y)}, Non-seizure: {len(y) - np.sum(y)}")

