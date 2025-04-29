from src.pose_extraction.mediapipe_extractor import MediaPipePoseExtractor
from src.data_processing.landmark_processor import landmarks_to_dataframe, save_landmarks

extractor = MediaPipePoseExtractor()
landmarks = extractor.extract_landmarks("path/to/video.mp4")
df = landmarks_to_dataframe(landmarks, model_name="mediapipe")
save_landmarks(df, "output_mediapipe.csv")
