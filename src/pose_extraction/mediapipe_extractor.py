import os
import cv2
import numpy as np
import mediapipe as mp

class MediapipePoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def _is_video_file(self, path):
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        return path.lower().endswith(video_extensions)

    def extract_landmarks(self, input_path):
        if input_path == "live":
            return self._extract_from_live_camera()
        elif os.path.isdir(input_path):
            return self._extract_from_frames_folder(input_path)
        elif self._is_video_file(input_path):
            return self._extract_from_video_file(input_path)
        else:
            raise ValueError(f"Unsupported input path: {input_path}")

    def _extract_from_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_landmarks = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            landmarks = self._process_frame(frame)
            all_landmarks.append(landmarks)

        cap.release()
        return np.array(all_landmarks, dtype=object)

    def _extract_from_frames_folder(self, folder_path):
        frame_files = sorted(os.listdir(folder_path))
        all_landmarks = []

        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            landmarks = self._process_frame(frame)
            all_landmarks.append(landmarks)

        return np.array(all_landmarks, dtype=object)

    def _extract_from_live_camera(self):
        cap = cv2.VideoCapture(0)  # Webcam input
        all_landmarks = []

        print("Press 'q' to quit live capture.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            landmarks = self._process_frame(frame)
            all_landmarks.append(landmarks)

            # Optional: Show live frame
            cv2.imshow('Live Pose Detection', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return np.array(all_landmarks, dtype=object)

    def _process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            return [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        else:
            return None
