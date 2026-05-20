import cv2
import numpy as np


LINE = "LINE"
CIRCLE = "CIRCLE"
RECTANGLE = "RECTANGLE"
TRIANGLE = "TRIANGLE"
UNKNOWN = "UNKNOWN"


class ShapeDetector:
    """Detects simple clean shapes from one completed drawing stroke."""

    def __init__(self, min_points=18, min_size=35, min_confidence=0.75):
        self.min_points = min_points
        self.min_size = min_size
        self.min_confidence = min_confidence

    def detect(self, points):
        """Return a shape result dictionary for the completed stroke."""
        if len(points) < self.min_points:
            return self._unknown()

        if self._looks_like_writing(points):
            return self._unknown()

        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        x, y, width, height = cv2.boundingRect(contour)

        if width < self.min_size and height < self.min_size:
            return self._unknown()

        line_result = self._detect_line(points)
        if line_result["confidence"] >= self.min_confidence:
            return line_result

        polygon_result = self._detect_polygon(contour)
        if polygon_result["confidence"] >= self.min_confidence:
            return polygon_result

        circle_result = self._detect_circle(points)
        if circle_result["confidence"] >= self.min_confidence:
            return circle_result

        return self._unknown()

    def _detect_line(self, points):
        """Classify a stroke as a line when points stay close to a fitted line."""
        point_array = np.array(points, dtype=np.float32)

        if len(point_array) < 2:
            return self._unknown()

        vx, vy, x0, y0 = cv2.fitLine(point_array, cv2.DIST_L2, 0, 0.01, 0.01)
        direction = np.array([float(vx[0]), float(vy[0])], dtype=np.float32)
        anchor = np.array([float(x0[0]), float(y0[0])], dtype=np.float32)

        distances = []
        projections = []
        for point in point_array:
            point_vector = point - anchor
            projection = float(np.dot(point_vector, direction))
            closest_point = anchor + projection * direction
            distances.append(float(np.linalg.norm(point - closest_point)))
            projections.append(projection)

        line_length = max(projections) - min(projections)
        average_distance = float(np.mean(distances))

        if line_length < self.min_size * 1.5:
            return self._unknown()

        distance_limit = max(7, line_length * 0.055)
        confidence = max(0.0, min(1.0, 1.0 - average_distance / distance_limit))

        if average_distance < distance_limit:
            start_values = np.round(anchor + min(projections) * direction).astype(int)
            end_values = np.round(anchor + max(projections) * direction).astype(int)
            start_point = (int(start_values[0]), int(start_values[1]))
            end_point = (int(end_values[0]), int(end_values[1]))
            return {
                "name": LINE,
                "confidence": confidence,
                "start": start_point,
                "end": end_point,
            }

        return self._unknown()

    def _detect_circle(self, points):
        """Classify a stroke as a circle when it is closed and radius is stable."""
        point_array = np.array(points, dtype=np.float32)
        (center_x, center_y), radius = cv2.minEnclosingCircle(point_array)

        if radius < self.min_size / 2:
            return self._unknown()

        start = point_array[0]
        end = point_array[-1]
        closed_distance = float(np.linalg.norm(start - end))
        close_ratio = closed_distance / radius
        if close_ratio > 0.42:
            return self._unknown()

        center = np.array([center_x, center_y], dtype=np.float32)
        distances = np.linalg.norm(point_array - center, axis=1)
        radius_error = float(np.std(distances) / radius)
        confidence = 1.0 - (radius_error / 0.24) * 0.7 - (close_ratio / 0.42) * 0.3

        if radius_error < 0.24:
            return {
                "name": CIRCLE,
                "confidence": max(0.0, min(1.0, confidence)),
                "center": (int(center_x), int(center_y)),
                "radius": int(radius),
            }

        return self._unknown()

    def _detect_polygon(self, contour):
        """Classify a closed stroke as a triangle or rectangle."""
        perimeter = cv2.arcLength(contour, closed=True)

        if perimeter < self.min_size * 3:
            return self._unknown()

        x, y, width, height = cv2.boundingRect(contour)
        start = contour[0][0].astype(np.float32)
        end = contour[-1][0].astype(np.float32)
        closed_distance = float(np.linalg.norm(start - end))
        diagonal = (width**2 + height**2) ** 0.5

        if diagonal == 0 or closed_distance > diagonal * 0.35:
            return self._unknown()

        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, closed=True)
        corners = [(int(point[0][0]), int(point[0][1])) for point in approx]
        area = abs(cv2.contourArea(contour))
        bounding_area = max(1, width * height)
        fill_ratio = area / bounding_area
        close_score = max(0.0, 1.0 - closed_distance / (diagonal * 0.35))

        if len(corners) == 3:
            confidence = max(0.0, min(1.0, 0.55 + 0.25 * close_score + 0.2 * min(1.0, fill_ratio / 0.45)))
            return {
                "name": TRIANGLE,
                "confidence": confidence,
                "points": corners,
            }

        if len(corners) == 4:
            if width >= self.min_size and height >= self.min_size:
                confidence = max(0.0, min(1.0, 0.55 + 0.25 * close_score + 0.2 * min(1.0, fill_ratio / 0.55)))
                return {
                    "name": RECTANGLE,
                    "confidence": confidence,
                    "rect": (x, y, width, height),
                }

        return self._unknown()

    def _unknown(self):
        return {"name": UNKNOWN, "confidence": 0.0}

    def _looks_like_writing(self, points):
        """Reject dense, loopy strokes that are more like writing than shapes."""
        if len(points) < 24:
            return False

        point_array = np.array(points, dtype=np.float32)
        contour = point_array.reshape((-1, 1, 2)).astype(np.int32)
        x, y, width, height = cv2.boundingRect(contour)
        diagonal = (width**2 + height**2) ** 0.5

        if diagonal < self.min_size:
            return True

        path_length = self._path_length(point_array)
        direct_distance = float(np.linalg.norm(point_array[-1] - point_array[0]))
        complexity = path_length / max(1.0, diagonal)

        # Writing often has lots of movement packed into a compact area.
        if complexity > 7.0 and direct_distance > diagonal * 0.25:
            return True

        return False

    def _path_length(self, point_array):
        distances = np.linalg.norm(point_array[1:] - point_array[:-1], axis=1)
        return float(np.sum(distances))
