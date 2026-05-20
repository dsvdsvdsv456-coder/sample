import cv2
import numpy as np

from utils.shape_detector import CIRCLE, LINE, RECTANGLE, UNKNOWN


PROCESS = "PROCESS"
START_END = "START/END"
CONNECTION = "CONNECTION"


class DiagramEngine:
    """Stores rough detected shapes and redraws them as a clean diagram."""

    def __init__(self):
        self.objects = []
        self.raw_connections = []
        self.connections = []
        self.base_canvas = None
        self.base_mask = None

    def reset(self):
        """Forget the current diagram model."""
        self.objects = []
        self.raw_connections = []
        self.connections = []
        self.base_canvas = None
        self.base_mask = None

    def ensure_base(self, drawing_canvas):
        """Remember the canvas before the first rough diagram stroke."""
        if self.base_canvas is None:
            self.base_canvas = drawing_canvas.canvas.copy()
            self.base_mask = drawing_canvas.mask.copy()

    def add_shape_result(self, shape_result):
        """Convert a detected shape into a diagram object or connector."""
        shape_name = shape_result.get("name", UNKNOWN)

        if shape_name == RECTANGLE:
            x, y, width, height = shape_result["rect"]
            self.objects.append(
                {
                    "type": PROCESS,
                    "kind": RECTANGLE,
                    "center": (x + width // 2, y + height // 2),
                    "size": (max(90, width), max(54, height)),
                }
            )
            return PROCESS

        if shape_name == CIRCLE:
            center = shape_result["center"]
            radius = max(32, shape_result["radius"])
            self.objects.append(
                {
                    "type": START_END,
                    "kind": CIRCLE,
                    "center": center,
                    "size": (radius * 2, radius * 2),
                }
            )
            return START_END

        if shape_name == LINE:
            self.raw_connections.append(
                {
                    "type": CONNECTION,
                    "start": shape_result["start"],
                    "end": shape_result["end"],
                }
            )
            return CONNECTION

        return UNKNOWN

    def object_count(self):
        """Return the number of diagram nodes, not connector lines."""
        return len(self.objects)

    def layout_and_draw(self, drawing_canvas):
        """Align objects and redraw the clean diagram with arrow connections."""
        if not self.objects:
            return False

        drawing_canvas.begin_action()

        if self.base_canvas is not None and self.base_mask is not None:
            drawing_canvas.canvas = self.base_canvas.copy()
            drawing_canvas.mask = self.base_mask.copy()

        self._layout_objects(drawing_canvas.canvas.shape)
        self._detect_connections()

        for diagram_object in self.objects:
            self._draw_object(drawing_canvas, diagram_object)

        for connection in self.connections:
            self._draw_arrow(drawing_canvas, connection["from"], connection["to"])

        drawing_canvas.commit_action()
        return True

    def _layout_objects(self, frame_shape):
        """Place objects in one clean horizontal row with equal spacing."""
        height, width = frame_shape[:2]
        object_count = len(self.objects)
        spacing = width // (object_count + 1)
        center_y = height // 2

        for index, diagram_object in enumerate(self.objects):
            diagram_object["center"] = (spacing * (index + 1), center_y)

    def _detect_connections(self):
        """Map rough connector lines to nearest two diagram objects."""
        self.connections = []

        if len(self.objects) < 2:
            return

        for raw_connection in self.raw_connections:
            start_object = self._nearest_object(raw_connection["start"])
            end_object = self._nearest_object(raw_connection["end"])

            if start_object is not None and end_object is not None and start_object is not end_object:
                self.connections.append({"from": start_object, "to": end_object})

        if not self.connections and len(self.objects) >= 2:
            for index in range(len(self.objects) - 1):
                self.connections.append({"from": self.objects[index], "to": self.objects[index + 1]})

    def _nearest_object(self, point):
        nearest = None
        best_distance = None

        for diagram_object in self.objects:
            distance = self._distance(point, diagram_object["center"])
            if best_distance is None or distance < best_distance:
                nearest = diagram_object
                best_distance = distance

        return nearest

    def _draw_object(self, drawing_canvas, diagram_object):
        color = drawing_canvas.color
        center_x, center_y = diagram_object["center"]
        width, height = diagram_object["size"]

        if diagram_object["kind"] == RECTANGLE:
            top_left = (center_x - width // 2, center_y - height // 2)
            bottom_right = (center_x + width // 2, center_y + height // 2)
            cv2.rectangle(drawing_canvas.canvas, top_left, bottom_right, color, drawing_canvas.brush_thickness, cv2.LINE_AA)
            cv2.rectangle(drawing_canvas.mask, top_left, bottom_right, 255, drawing_canvas.brush_thickness, cv2.LINE_AA)
            return

        radius = max(width, height) // 2
        cv2.circle(drawing_canvas.canvas, (center_x, center_y), radius, color, drawing_canvas.brush_thickness, cv2.LINE_AA)
        cv2.circle(drawing_canvas.mask, (center_x, center_y), radius, 255, drawing_canvas.brush_thickness, cv2.LINE_AA)

    def _draw_arrow(self, drawing_canvas, from_object, to_object):
        color = drawing_canvas.color
        start = self._edge_point(from_object, to_object)
        end = self._edge_point(to_object, from_object)

        cv2.arrowedLine(
            drawing_canvas.canvas,
            start,
            end,
            color,
            drawing_canvas.brush_thickness,
            cv2.LINE_AA,
            tipLength=0.16,
        )
        cv2.arrowedLine(
            drawing_canvas.mask,
            start,
            end,
            255,
            drawing_canvas.brush_thickness,
            cv2.LINE_AA,
            tipLength=0.16,
        )

    def _edge_point(self, source_object, target_object):
        source_x, source_y = source_object["center"]
        target_x, target_y = target_object["center"]
        width, height = source_object["size"]

        vector = np.array([target_x - source_x, target_y - source_y], dtype=np.float32)
        length = float(np.linalg.norm(vector))
        if length == 0:
            return source_object["center"]

        unit = vector / length
        radius_x = width / 2
        radius_y = height / 2
        scale = min(
            abs(radius_x / unit[0]) if unit[0] != 0 else float("inf"),
            abs(radius_y / unit[1]) if unit[1] != 0 else float("inf"),
        )

        point = np.array([source_x, source_y], dtype=np.float32) + unit * scale
        return (int(point[0]), int(point[1]))

    def _distance(self, first_point, second_point):
        first_x, first_y = first_point
        second_x, second_y = second_point
        return ((second_x - first_x) ** 2 + (second_y - first_y) ** 2) ** 0.5
