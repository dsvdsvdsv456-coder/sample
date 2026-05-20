from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    """Small beginner-friendly wrapper around MediaPipe Tasks HandLandmarker."""

    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        max_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.model_path = Path(model_path)
        self.results = None

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Missing MediaPipe hand model: models/hand_landmarker.task\n"
                "Run this command first:\n"
                "python download_model.py"
            )

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        """Detect hand landmarks in a BGR OpenCV frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return self.results

    def get_landmarks(self, frame):
        """Return the first hand as a list of (id, x, y) pixel coordinates."""
        landmark_list = []

        if not self.results or not self.results.hand_landmarks:
            return landmark_list

        frame_height, frame_width, _ = frame.shape
        first_hand = self.results.hand_landmarks[0]

        for landmark_id, landmark in enumerate(first_hand):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            landmark_list.append((landmark_id, x, y))

        return landmark_list

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
