import cv2
import mediapipe as mp
import numpy as np

class MediaPipePoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                                           model_complexity=2,
                                           enable_segmentation=False,
                                           min_detection_confidence=0.5)

    def extract_landmarks(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_landmarks = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                all_landmarks.append(landmarks)
            else:
                all_landmarks.append(None)

        cap.release()
        return np.array(all_landmarks, dtype=object)  # object dtype because of None frames
