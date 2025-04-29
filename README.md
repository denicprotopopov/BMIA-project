# Epileptic Seizure Pose Extraction

Extraction of 3D human pose landmarks using:
- MediaPipe Pose
- MoveNet Thunder

## Structure
- `src/pose_extraction/`: Landmark extractors
- `src/data_processing/`: Landmark post-processing
- `examples/`: Example extraction scripts



# How to Run Pose Extraction Scripts

This project extracts 3D pose landmarks using either **MediaPipe** or **MoveNet**, from:
- Video files
- Folders of frames
- Live webcam streams

Extracted landmarks are saved as `.csv` files in the `data/DATASET_NAME/processed/` folder.

---

## 1. Setup (Only Once)

Before running any scripts:

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd <your-repo-name>

# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# OR
source .venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## 3. Input Data Types Supported

The code supports three types of input:

| Type | Description |
|:-----|:------------|
| Video file | `.mp4`, `.avi`, `.mov`, `.mkv` formats |
| Folder of frames | A directory containing sequential `.jpg` or `.png` images |
| Live webcam | Real-time pose extraction from the default webcam |

---

## 4. Running Pose Extraction

Step 1: Place your video file or frame folder inside `data/raw/`.

Step 2: Edit the `input_path` in the chosen example script.

- For MediaPipe:

Edit `examples/extract_with_mediapipe.py`:
```bash
input_path = "data/raw/your_video_file.mp4"
```

Run:
```bash
python examples/extract_with_mediapipe.py
```
- For MoveNet:

Edit `examples/extract_with_movenet.py`:
```bash
input_path = "data/raw/your_video_file.mp4"
```

Run:
```bash
python examples/extract_with_movenet.py
```


Output:  
A `.csv` file will be saved in `data/processed/` with the input file's name and model used.

---

For live video (input_path = "live"):  
The webcam will open, showing a live feed with pose detection.  
Press `q` to quit the live session.  
The extracted landmarks will be saved as a CSV in `data/processed/`.

---

## 5. Output File Details

- Extracted landmark files are saved as `.csv` files.
- They are located inside the `data/processed/` directory.
- Filenames are automatically generated based on the source input and model used.

Examples:

| Input | Model | Output Filename |
|:------|:------|:-----------------|
| `your_video_file.mp4` | MediaPipe | `your_video_file_mediapipe.csv` |
| `your_frames_folder/` | MoveNet | `your_frames_folder_movenet.csv` |
| Live Webcam | MediaPipe | `live_mediapipe.csv` |

Each row in the CSV corresponds to one frame, and each column represents the flattened 3D coordinates (x, y, z) of the pose landmarks.

---

## 6. Additional Notes

- Always activate your `.venv` before running any scripts.
- If processing live video, a webcam must be properly connected and accessible.
- The `data/processed/` folder is created automatically if it does not exist.
- Scripts automatically detect whether the input is a video file, folder of frames, or live webcam.

---

## 7. Checklist Before Running

| Step | Check |
|:-----|:------|
| Virtual environment activated (`.venv`) | ✅ |
| Dependencies installed from `requirements.txt` | ✅ |
| `input_path` correctly set in the example script | ✅ |
| Using the correct terminal (Git Bash recommended) | ✅ |
| Input data properly placed in `data/raw/` | ✅ |
| Webcam functional for live capture (if needed) | ✅ |

---