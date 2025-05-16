import os
import sys
<<<<<<< Updated upstream
<<<<<<< Updated upstream
# Add the project root to sys.path
=======

# Add the src directory to the Python path
>>>>>>> Stashed changes
=======

# Add the src directory to the Python path
>>>>>>> Stashed changes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pose_extraction.movenet_extractor import MoveNetPoseExtractor
from src.data_processing.landmark_processor import landmarks_to_dataframe, save_landmarks

# Set input
<<<<<<< Updated upstream
<<<<<<< Updated upstream
input_path = "data/3DPW/raw/imageFiles/imageFiles/outdoors_slalom_00"    # <-- Update this path
=======
input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM706-2_sz03_kinect.mp4"  # <-- Update this path
>>>>>>> Stashed changes
=======
input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM706-2_sz03_kinect.mp4"  # <-- Update this path
>>>>>>> Stashed changes

# Extract
extractor = MoveNetPoseExtractor()
landmarks = extractor.extract_landmarks(input_path)
df = landmarks_to_dataframe(landmarks, model_name="movenet")

# Prepare output path
output_folder = "data/keypoints/" # <-- Update this path
os.makedirs(output_folder, exist_ok=True)

# Create a safe identifiable output filename
input_name = os.path.basename(input_path.rstrip("/")).split('.')[0]
output_filename = f"{input_name}_movenet.csv"
output_path = os.path.join(output_folder, output_filename)

# Save
save_landmarks(df, output_path)
print(f"Saved landmarks to {output_path}")
