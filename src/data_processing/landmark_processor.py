import numpy as np
import pandas as pd

def landmarks_to_dataframe(landmarks_sequence, model_name="mediapipe"):
    frames = []
    for idx, landmarks in enumerate(landmarks_sequence):
        if landmarks is not None:
            flat_landmarks = np.array(landmarks).flatten()
        else:
            flat_landmarks = np.full((33*3 if model_name=="mediapipe" else 17*3,), np.nan)
        
        frames.append(flat_landmarks)
    
    df = pd.DataFrame(frames)
    return df

def save_landmarks(df, output_path):
    df.to_csv(output_path, index=False)
