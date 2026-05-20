class GestureDetector:
    """Detects simple hand gestures from HandLandmarker pixel landmarks."""

    AIM = "AIM"
    SELECT_MODE = "SELECT_MODE"
    IDLE = "IDLE"

    # Backward-friendly names for older code paths.
    PEN_DRAW = AIM
    DRAW = AIM
    ERASER = AIM
    SELECT = SELECT_MODE

    def __init__(self, min_hand_confidence=0.55):
        self.min_hand_confidence = min_hand_confidence

    def get_mode(self, landmarks, hand_result=None):
        """Return AIM, SELECT_MODE, or IDLE from the current hand pose."""
        if not landmarks:
            return self.IDLE

        if not self._has_strong_hand_confidence(hand_result):
            return self.IDLE

        points = {landmark_id: (x, y) for landmark_id, x, y in landmarks}

        required_landmarks = [0, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20]
        if not all(landmark_id in points for landmark_id in required_landmarks):
            return self.IDLE

        index_up = self._finger_is_up(points, tip_id=8, pip_id=6)
        middle_up = self._finger_is_up(points, tip_id=12, pip_id=10)
        ring_up = self._finger_is_up(points, tip_id=16, pip_id=14)
        pinky_up = self._finger_is_up(points, tip_id=20, pip_id=18)

        if index_up and middle_up and not ring_up and not pinky_up:
            return self.SELECT_MODE

        if index_up:
            return self.AIM

        if not index_up and not middle_up and not ring_up and not pinky_up:
            return self.IDLE

        # Open palm or other visible hand poses are preview-only.
        return self.AIM

    def _finger_is_up(self, points, tip_id, pip_id):
        """A finger is up when its tip is higher than its middle joint."""
        tip_y = points[tip_id][1]
        pip_y = points[pip_id][1]

        # In image coordinates, smaller y values are higher on the screen.
        return tip_y < pip_y

    def is_index_only(self, landmarks):
        """Return True only when index is up and the other fingers are down."""
        if not landmarks:
            return False

        points = {landmark_id: (x, y) for landmark_id, x, y in landmarks}
        required_landmarks = [6, 8, 10, 12, 14, 16, 18, 20]

        if not all(landmark_id in points for landmark_id in required_landmarks):
            return False

        index_up = self._finger_is_up(points, tip_id=8, pip_id=6)
        middle_up = self._finger_is_up(points, tip_id=12, pip_id=10)
        ring_up = self._finger_is_up(points, tip_id=16, pip_id=14)
        pinky_up = self._finger_is_up(points, tip_id=20, pip_id=18)

        return index_up and not middle_up and not ring_up and not pinky_up

    def _has_strong_hand_confidence(self, hand_result):
        """Use HandLandmarker handedness confidence when it is available."""
        if hand_result is None:
            return True

        if not hand_result.hand_landmarks:
            return False

        if not hand_result.handedness or not hand_result.handedness[0]:
            return False

        best_handedness = hand_result.handedness[0][0]
        return best_handedness.score >= self.min_hand_confidence
