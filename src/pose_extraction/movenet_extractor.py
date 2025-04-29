import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class MoveNetPoseExtractor:
    def __init__(self):
        model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        self.model = hub.load(model_url)

    def extract_landmarks(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_landmarks = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            input_image = cv2.resize(frame, (256, 256))
            input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)[tf.newaxis, ...]

            outputs = self.model.signatures['serving_default'](input_image)
            keypoints = outputs['output_0'].numpy()[0, 0, :, :]

            all_landmarks.append(keypoints)  # shape: (17, 3)

        cap.release()
        return np.array(all_landmarks, dtype=object)
