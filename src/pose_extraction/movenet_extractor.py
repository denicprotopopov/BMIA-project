import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class MoveNetPoseExtractor:
    def __init__(self):
        model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        self.model = hub.load(model_url)

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
        cap = cv2.VideoCapture(0)  # Webcam
        all_landmarks = []

        print("Press 'q' to quit live capture.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            landmarks = self._process_frame(frame)
            all_landmarks.append(landmarks)

            # Optional: show live feed
            cv2.imshow('Live Pose Detection (MoveNet)', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return np.array(all_landmarks, dtype=object)

    def _process_frame(self, frame):
        # Preprocessing
        input_image = cv2.resize(frame, (256, 256))
        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image = tf.expand_dims(input_image, axis=0)

        # Inference
        outputs = self.model.signatures['serving_default'](input_image)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # shape: (17, 3)

        return keypoints
