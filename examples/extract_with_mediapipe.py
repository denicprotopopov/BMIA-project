import os
import sys
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pose_extraction.mediapipe_extractor import MediapipePoseExtractor
from src.data_processing.landmark_processor import landmarks_to_dataframe, save_landmarks

# Set input
<<<<<<< Updated upstream

# input_path = "data/3DPW/raw/imageFiles/imageFiles/outdoors_slalom_00"  # <-- Update this path

input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM722-4_sz06_kinect.mp4"  # <-- Update this path
=======
input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM706-2_sz03_kinect.mp4"  # <-- Update this path
>>>>>>> Stashed changes

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
