import cv2


CLEAN_MODE = "CLEAN"
AI_MODE = "AI"
DEMO_MODE = "DEMO"
VISUAL_MODES = [CLEAN_MODE, AI_MODE, DEMO_MODE]


class VisualOverlay:
    """Draws optional hand visualization layers for demos."""

    TECHNICAL_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    DEMO_CONNECTIONS = [
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (1, 2),
        (2, 3),
        (3, 4),
    ]

    IMPORTANT_LANDMARKS = [0, 4, 8, 12]

    def draw_technical_landmarks(self, frame, landmarks):
        """Draw the full hand skeleton for debugging or AI mode."""
        points = self._to_points(landmarks)
        if not points:
            return

        for start_id, end_id in self.TECHNICAL_CONNECTIONS:
            if start_id in points and end_id in points:
                cv2.line(frame, points[start_id], points[end_id], (90, 210, 175), 1, cv2.LINE_AA)

        for point in points.values():
            cv2.circle(frame, point, 3, (240, 245, 248), -1, cv2.LINE_AA)

    def draw_demo_overlay(self, frame, landmarks, status, cursor_point):
        """Draw a premium lightweight gesture overlay for public demos."""
        points = self._to_points(landmarks)
        if not points:
            return

        for start_id, end_id in self.DEMO_CONNECTIONS:
            if start_id in points and end_id in points:
                cv2.line(frame, points[start_id], points[end_id], (92, 220, 210), 1, cv2.LINE_AA)

        for landmark_id in self.IMPORTANT_LANDMARKS:
            if landmark_id in points:
                self._draw_glow_dot(frame, points[landmark_id])

        label_point = cursor_point or points.get(8)
        if label_point is not None:
            self._draw_gesture_label(frame, label_point, status)

    def _draw_glow_dot(self, frame, point):
        """Draw a small glowing dot by layering soft circles."""
        cv2.circle(frame, point, 10, (60, 180, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, point, 6, (80, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 2, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_gesture_label(self, frame, point, status):
        """Place a compact gesture label near the hand."""
        x, y = point
        x = min(max(12, x + 18), frame.shape[1] - 210)
        y = min(max(90, y - 34), frame.shape[0] - 70)

        hint = self._hint_for_status(status)
        label = "HOLDING" if status == "HOLDING..." else status
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 248, 250), 1, cv2.LINE_AA)
        cv2.putText(frame, hint, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (170, 230, 255), 1, cv2.LINE_AA)

    def _hint_for_status(self, status):
        if status == "DRAW_LOCK":
            return "Drawing locked"

        if status == "SELECT/PAUSE":
            return "Two fingers = pause"

        return "Hold to draw"

    def _to_points(self, landmarks):
        return {landmark_id: (x, y) for landmark_id, x, y in landmarks}
