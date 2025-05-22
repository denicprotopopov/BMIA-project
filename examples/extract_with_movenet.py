import os
import sys
import time 

start_time = time.time()
# Add the project root to sys.path
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pose_extraction.movenet_extractor import MoveNetPoseExtractor
from src.data_processing.landmark_processor import landmarks_to_dataframe, save_landmarks

# Set input
# input_path = "data/3DPW/raw/imageFiles/imageFiles/outdoors_slalom_00"    # <-- Update this path
# 1 
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM706-2_sz03_kinect.mp4"  # <-- Update this path
# 2
input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM722-4_sz06_kinect.mp4"  # <-- Update this path
# 3
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1183-2_sz12_kinect.mp4"  # <-- Update this path
# 4
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1239_sz07_kinect.mp4"  # <-- Update this path
# 5
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1267_sz10_kinect.mp4"  # <-- Update this path
# 6 
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1279_sz06_kinect.mp4"  # <-- Update this path
# 7     
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1372_sz06_kinect.mp4"  # <-- Update this path
# 8
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1405_sz03_kinect.mp4"  # <-- Update this path
# 9
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1540_sz02_kinect.mp4"  # <-- Update this path
# 10
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM646-4_sz04_kinect.mp4"  # <-- Update this path
# 11
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM719-2_sz04_kinect.mp4"  # <-- Update this path
# 12
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM722-4_sz01_kinect.mp4"  # <-- Update this path
# 13
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM790-2_sz01_kinect.mp4"  # <-- Update this path
# 14
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM790-2_sz03_kinect.mp4"  # <-- Update this path
# 15
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1166-2_sz23_kinect.mp4"  # <-- Update this path
# 16
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1263_sz03_kinect.mp4"  # <-- Update this path
# 17
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1321_sz02_kinect.mp4"  # <-- Update this path
# 18 
# input_path = r"D:\JPC_Datasets(obsolete)\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset\BioImAnalysis_project_dataset_VIDEOS\automotor_MP4\IM1321_sz04_kinect.mp4"  # <-- Update this path


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

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")