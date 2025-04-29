from src.pose_extraction.movenet_extractor import MoveNetPoseExtractor
from src.data_processing.landmark_processor import landmarks_to_dataframe, save_landmarks

extractor = MoveNetPoseExtractor()
landmarks = extractor.extract_landmarks("path/to/video.mp4")
df = landmarks_to_dataframe(landmarks, model_name="movenet")
save_landmarks(df, "output_movenet.csv")
