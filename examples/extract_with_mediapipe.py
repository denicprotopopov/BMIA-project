import os
from src.pose_extraction.mediapipe_extractor import MediapipePoseExtractor
from src.data_processing.landmark_processor import landmarks_to_dataframe, save_landmarks

# Set input
input_path = "data/3DPW/raw/imageFiles/downtown_cafe_00"  # <-- Update this path

# Extract
extractor = MediapipePoseExtractor()
landmarks = extractor.extract_landmarks(input_path)
df = landmarks_to_dataframe(landmarks, model_name="mediapipe")

# Prepare output path
output_folder = "data/3DPW/processed/" # <-- Update this path
os.makedirs(output_folder, exist_ok=True)  # Create folder if not exists

# Create a safe identifiable output filename
input_name = os.path.basename(input_path.rstrip("/")).split('.')[0]  # folder or file name
output_filename = f"{input_name}_mediapipe.csv"
output_path = os.path.join(output_folder, output_filename)

# Save
save_landmarks(df, output_path)
print(f"Saved landmarks to {output_path}")
