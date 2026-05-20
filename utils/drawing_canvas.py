from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from utils.shape_detector import CIRCLE, LINE, RECTANGLE, TRIANGLE


class DrawingCanvas:
    """Stores and draws the user's air drawing on a transparent-style canvas."""

    def __init__(
        self,
        frame_shape,
        color=(0, 0, 255),
        brush_thickness=5,
        eraser_thickness=30,
        smoothing_alpha=0.45,
        min_movement=5,
        max_jump=80,
        no_draw_top=50,
    ):
        self.frame_shape = frame_shape
        self.color = color
        self.brush_thickness = brush_thickness
        self.eraser_thickness = eraser_thickness
        self.smoothing_alpha = smoothing_alpha
        self.min_movement = min_movement
        self.max_jump = max_jump
        self.no_draw_top = no_draw_top
        self.previous_point = None
        self.smoothed_point = None
        self.stroke_backup = None
        self.mask_backup = None
        self.action_backup = None
        self.action_mask_backup = None
        self.undo_stack = []
        self.redo_stack = []
        self.canvas = self._create_blank_canvas(frame_shape)
        self.mask = self._create_blank_mask(frame_shape)

    def _create_blank_canvas(self, frame_shape):
        """Create a black canvas with the same height and width as the webcam."""
        height, width = frame_shape[:2]
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _create_blank_mask(self, frame_shape):
        """Create a mask that marks where drawing pixels exist."""
        height, width = frame_shape[:2]
        return np.zeros((height, width), dtype=np.uint8)

    def resize_if_needed(self, frame_shape):
        """Recreate the canvas if the webcam frame size changes."""
        if self.canvas.shape[:2] != frame_shape[:2]:
            self.frame_shape = frame_shape
            self.canvas = self._create_blank_canvas(frame_shape)
            self.mask = self._create_blank_mask(frame_shape)
            self.reset_previous_point()

    def begin_action(self):
        """Save the canvas before one stroke/action so undo can restore it."""
        self.action_backup = self.canvas.copy()
        self.action_mask_backup = self.mask.copy()

    def commit_action(self):
        """Store the saved pre-action state if the canvas changed."""
        if self.action_backup is None or self.action_mask_backup is None:
            return

        canvas_changed = not np.array_equal(self.canvas, self.action_backup)
        mask_changed = not np.array_equal(self.mask, self.action_mask_backup)

        if canvas_changed or mask_changed:
            self.undo_stack.append((self.action_backup, self.action_mask_backup))
            self.redo_stack.clear()

        self.action_backup = None
        self.action_mask_backup = None

    def discard_action(self):
        """Forget a saved action when nothing useful happened."""
        self.action_backup = None
        self.action_mask_backup = None

    def start_stroke(self, current_point):
        """Prepare for pen or eraser movement without drawing a first jump line."""
        smoothed_point = self.smooth_point(current_point)
        self.previous_point = smoothed_point
        return smoothed_point

    def begin_smart_stroke(self):
        """Remember the canvas before a rough stroke starts."""
        self.stroke_backup = self.canvas.copy()
        self.mask_backup = self.mask.copy()

    def restore_smart_stroke_backup(self):
        """Remove the temporary rough stroke by restoring the saved canvas."""
        if self.stroke_backup is not None:
            self.canvas = self.stroke_backup.copy()
        if self.mask_backup is not None:
            self.mask = self.mask_backup.copy()

    def clear_smart_stroke_backup(self):
        """Forget the saved pre-stroke canvas."""
        self.stroke_backup = None
        self.mask_backup = None

    def draw_detected_shape(self, shape_result):
        """Draw a clean detected shape onto the canvas."""
        shape_name = shape_result.get("name")

        if shape_name == LINE:
            cv2.line(
                self.canvas,
                shape_result["start"],
                shape_result["end"],
                self.color,
                self.brush_thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                self.mask,
                shape_result["start"],
                shape_result["end"],
                255,
                self.brush_thickness,
                cv2.LINE_AA,
            )
        elif shape_name == CIRCLE:
            cv2.circle(
                self.canvas,
                shape_result["center"],
                shape_result["radius"],
                self.color,
                self.brush_thickness,
                cv2.LINE_AA,
            )
            cv2.circle(
                self.mask,
                shape_result["center"],
                shape_result["radius"],
                255,
                self.brush_thickness,
                cv2.LINE_AA,
            )
        elif shape_name == RECTANGLE:
            x, y, width, height = shape_result["rect"]
            cv2.rectangle(
                self.canvas,
                (x, y),
                (x + width, y + height),
                self.color,
                self.brush_thickness,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                self.mask,
                (x, y),
                (x + width, y + height),
                255,
                self.brush_thickness,
                cv2.LINE_AA,
            )
        elif shape_name == TRIANGLE:
            points = np.array(shape_result["points"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                self.canvas,
                [points],
                isClosed=True,
                color=self.color,
                thickness=self.brush_thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.polylines(
                self.mask,
                [points],
                isClosed=True,
                color=255,
                thickness=self.brush_thickness,
                lineType=cv2.LINE_AA,
            )

    def draw_text(self, text, position, font_scale, color):
        """Draw text onto the canvas and update the drawing mask."""
        if not text:
            return

        thickness = max(1, int(font_scale * 2))
        cv2.putText(
            self.canvas,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            self.mask,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            255,
            thickness,
            cv2.LINE_AA,
        )

    def draw_pen(self, current_point):
        """Draw a colored pen stroke from the previous point to the current one."""
        return self._draw_stroke(
            current_point=current_point,
            color=self.color,
            thickness=self.brush_thickness,
        )

    def draw_eraser(self, current_point):
        """Erase strokes by removing pixels from the drawing mask."""
        return self._draw_stroke(
            current_point=current_point,
            color=(0, 0, 0),
            thickness=self.eraser_thickness,
            erase=True,
        )

    def draw(self, current_point):
        """Backward-friendly alias for normal pen drawing."""
        return self.draw_pen(current_point)

    def _draw_stroke(self, current_point, color, thickness, erase=False):
        """Draw from the previous point to the smoothed current finger point."""
        if current_point is None:
            self.reset_previous_point()
            return None

        smoothed_point = self.smooth_point(current_point)

        if smoothed_point[1] <= self.no_draw_top:
            self.previous_point = smoothed_point
            return smoothed_point

        if self.previous_point is None:
            self.previous_point = smoothed_point
            return smoothed_point

        if self._distance(self.previous_point, smoothed_point) > self.max_jump:
            self.previous_point = smoothed_point
            return smoothed_point

        if self._distance(self.previous_point, smoothed_point) < self.min_movement:
            return smoothed_point

        if erase:
            cv2.line(
                self.canvas,
                self.previous_point,
                smoothed_point,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.line(
                self.mask,
                self.previous_point,
                smoothed_point,
                0,
                thickness,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.line(
                self.canvas,
                self.previous_point,
                smoothed_point,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.line(
                self.mask,
                self.previous_point,
                smoothed_point,
                255,
                thickness,
                lineType=cv2.LINE_AA,
            )
        self.previous_point = smoothed_point
        return smoothed_point

    def set_color(self, color):
        """Change the current pen color."""
        self.color = color

    def increase_brush_size(self, amount=1, max_size=40):
        """Make the pen brush larger."""
        self.brush_thickness = min(max_size, self.brush_thickness + amount)

    def decrease_brush_size(self, amount=1, min_size=1):
        """Make the pen brush smaller."""
        self.brush_thickness = max(min_size, self.brush_thickness - amount)

    def increase_eraser_size(self, amount=2, max_size=80):
        """Make the eraser larger."""
        self.eraser_thickness = min(max_size, self.eraser_thickness + amount)

    def decrease_eraser_size(self, amount=2, min_size=8):
        """Make the eraser smaller."""
        self.eraser_thickness = max(min_size, self.eraser_thickness - amount)

    def smooth_point(self, current_point):
        """Smooth the fingertip point with an exponential moving average."""
        if self.smoothed_point is None:
            self.smoothed_point = current_point
            return current_point

        current_x, current_y = current_point
        previous_x, previous_y = self.smoothed_point

        smoothed_x = (
            self.smoothing_alpha * current_x
            + (1 - self.smoothing_alpha) * previous_x
        )
        smoothed_y = (
            self.smoothing_alpha * current_y
            + (1 - self.smoothing_alpha) * previous_y
        )

        self.smoothed_point = (int(smoothed_x), int(smoothed_y))
        return self.smoothed_point

    def _distance(self, first_point, second_point):
        """Measure pixel distance between two points."""
        first_x, first_y = first_point
        second_x, second_y = second_point
        return ((second_x - first_x) ** 2 + (second_y - first_y) ** 2) ** 0.5

    def reset_previous_point(self):
        """Stop connecting lines when drawing pauses."""
        self.previous_point = None
        self.smoothed_point = None

    def clear_canvas(self):
        """Erase everything currently drawn."""
        self.canvas[:] = 0
        self.mask[:] = 0
        self.reset_previous_point()

    def undo(self):
        """Restore the canvas to the state before the last action."""
        if not self.undo_stack:
            return False

        self.redo_stack.append((self.canvas.copy(), self.mask.copy()))
        self.canvas, self.mask = self.undo_stack.pop()
        self.reset_previous_point()
        return True

    def redo(self):
        """Re-apply the most recently undone action."""
        if not self.redo_stack:
            return False

        self.undo_stack.append((self.canvas.copy(), self.mask.copy()))
        self.canvas, self.mask = self.redo_stack.pop()
        self.reset_previous_point()
        return True

    def save_canvas(self, folder_path="assets/saved_drawings"):
        """Save the current canvas image and return the saved file path."""
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = folder / f"drawing_{timestamp}.png"
        cv2.imwrite(str(file_path), self.render_on_background((255, 255, 255)))
        return file_path

    def save_canvas_for_page(
        self,
        page_number,
        folder_path="assets/saved_drawings",
        background_color=(255, 255, 255),
    ):
        """Save the current page image with the page number in the filename."""
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = folder / f"drawing_page_{page_number}_{timestamp}.png"
        cv2.imwrite(str(file_path), self.render_on_background(background_color))
        return file_path

    def render_on_background(self, background_color):
        """Render the drawing on a solid board background."""
        background = np.full(self.canvas.shape, background_color, dtype=np.uint8)
        return self._compose_over(background)

    def overlay_on_frame(self, frame):
        """Overlay the drawing canvas on top of the webcam frame."""
        self.resize_if_needed(frame.shape)

        return self._compose_over(frame)

    def _compose_over(self, background):
        """Composite drawing pixels over a camera frame or board background."""
        drawing_mask = self.mask
        drawing_mask_inv = cv2.bitwise_not(drawing_mask)

        frame_background = cv2.bitwise_and(background, background, mask=drawing_mask_inv)
        drawing_foreground = cv2.bitwise_and(
            self.canvas,
            self.canvas,
            mask=drawing_mask,
        )

        return cv2.add(frame_background, drawing_foreground)
