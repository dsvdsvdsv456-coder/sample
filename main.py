import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from utils.diagram_engine import DiagramEngine
from utils.gesture_detector import GestureDetector
from utils.hand_tracker import HandTracker
from utils.page_manager import PageManager
from utils.shape_detector import UNKNOWN, ShapeDetector
from utils.smoothing import ExponentialPointSmoother
from utils.text_tool import TextTool
from utils.ui_toolbar import UIToolbar
from utils.visual_overlay import AI_MODE, CLEAN_MODE, DEMO_MODE, VISUAL_MODES, VisualOverlay


WINDOW_NAME = "Touchless AI Drawing Studio"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
HOLD_SECONDS = 0.6
HOLD_RADIUS = 14
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

COLORS = {
    ord("1"): ("RED", (38, 54, 235)),
    ord("2"): ("GREEN", (73, 190, 92)),
    ord("3"): ("BLUE", (238, 132, 67)),
    ord("4"): ("YELLOW", (42, 210, 245)),
    ord("5"): ("WHITE", (245, 245, 245)),
}

PEN_TOOL = "PEN"
ERASER_TOOL = "ERASER"
SHAPE_TOOL = "SHAPE"
TEXT_TOOL = "TEXT"
CAMERA_MODE = "Camera"
WHITEBOARD_MODE = "Whiteboard"
WHITE_BOARD = "White"
DARK_BOARD = "Dark"


class HoldInteractionEngine:
    """Turns an aiming cursor into drawing only after a steady hold."""

    def __init__(self, hold_seconds=0.6, hold_radius=14):
        self.hold_seconds = hold_seconds
        self.hold_radius = hold_radius
        self.hold_start_time = None
        self.hold_anchor = None
        self.draw_locked = False
        self.just_locked = False

    def update(self, gesture_mode, cursor_point, is_index_only):
        """Update the interaction state for the current frame."""
        self.just_locked = False
        now = time.monotonic()

        if cursor_point is None or gesture_mode == GestureDetector.IDLE:
            self.stop()
            return GestureDetector.IDLE, 0.0

        if gesture_mode == GestureDetector.SELECT_MODE:
            self.stop()
            return "SELECT/PAUSE", 0.0

        if self.draw_locked:
            return "DRAW_LOCK", 1.0

        if not is_index_only:
            self._reset_hold()
            return "AIM", 0.0

        if self.hold_anchor is None:
            self.hold_anchor = cursor_point
            self.hold_start_time = now
            return "HOLDING...", 0.0

        if self._distance(self.hold_anchor, cursor_point) > self.hold_radius:
            self.hold_anchor = cursor_point
            self.hold_start_time = now
            return "HOLDING...", 0.0

        hold_progress = min(1.0, (now - self.hold_start_time) / self.hold_seconds)

        if hold_progress >= 1.0:
            self.draw_locked = True
            self.just_locked = True
            return "DRAW_LOCK", 1.0

        return "HOLDING...", hold_progress

    def stop(self):
        """Stop drawing and clear any pending hold."""
        self.draw_locked = False
        self._reset_hold()

    def _reset_hold(self):
        self.hold_start_time = None
        self.hold_anchor = None
        self.just_locked = False

    def _distance(self, first_point, second_point):
        first_x, first_y = first_point
        second_x, second_y = second_point
        return ((second_x - first_x) ** 2 + (second_y - first_y) ** 2) ** 0.5


def resize_with_aspect_ratio(frame, target_width, target_height):
    """Fit the webcam frame inside the target size without stretching it."""
    frame_height, frame_width = frame.shape[:2]
    scale = min(target_width / frame_width, target_height / frame_height)

    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    output = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    mapping = {
        "scale": scale,
        "offset_x": left,
        "offset_y": top,
        "content_width": new_width,
        "content_height": new_height,
    }
    return output, mapping


def sharpen_frame(frame):
    """Apply a mild sharpening pass so the camera looks a little clearer."""
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.25, blurred, -0.25, 0)


def configure_camera(cap):
    """Ask the webcam for better capture settings when the driver supports it."""
    requested_settings = [
        (cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH),
        (cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT),
        (cv2.CAP_PROP_FPS, CAMERA_FPS),
        (cv2.CAP_PROP_BRIGHTNESS, 0.55),
        (cv2.CAP_PROP_CONTRAST, 0.55),
        (cv2.CAP_PROP_SHARPNESS, 0.55),
    ]

    for property_id, value in requested_settings:
        cap.set(property_id, value)


def enhance_frame(frame):
    """Apply a light camera enhancement pipeline for a cleaner demo image."""
    # 1. Light denoising. Gaussian blur is fast and removes small color noise.
    denoised = cv2.GaussianBlur(frame, (3, 3), 0)

    # 2. Contrast enhancement on the L channel keeps color changes controlled.
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    contrast_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 3. Slight gamma lift brightens midtones without blowing out highlights.
    gamma_corrected = apply_gamma(contrast_frame, gamma=0.95)

    # 4. Small saturation boost makes the image less dull.
    hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    s_channel = cv2.convertScaleAbs(s_channel, alpha=1.08, beta=0)
    saturated = cv2.cvtColor(cv2.merge((h_channel, s_channel, v_channel)), cv2.COLOR_HSV2BGR)

    # 5. Final sharpening kernel restores edge crispness after denoising.
    sharpening_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    return cv2.filter2D(saturated, -1, sharpening_kernel)


def apply_gamma(frame, gamma=1.0):
    """Apply gamma correction with a small lookup table for speed."""
    inverse_gamma = 1.0 / gamma
    table = np.array(
        [((value / 255.0) ** inverse_gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame, table)


def draw_cursor(frame, point, color, status, hold_progress):
    """Draw the fingertip cursor, hold progress ring, and draw-lock glow."""
    if point is None:
        return

    if status == "DRAW_LOCK":
        cv2.circle(frame, point, 15, color, 1, cv2.LINE_AA)
        cv2.circle(frame, point, 9, color, 2, cv2.LINE_AA)
        return

    cv2.circle(frame, point, 8, color, 2, cv2.LINE_AA)

    if status == "HOLDING...":
        end_angle = int(360 * hold_progress)
        cv2.ellipse(frame, point, (16, 16), -90, 0, end_angle, color, 2, cv2.LINE_AA)


def handle_key(key, drawing_canvas, active_tool, color_name, smart_shape_enabled):
    """Handle normal keyboard controls and return updated UI state."""
    message = ""

    if key in (ord("c"), ord("C")):
        drawing_canvas.begin_action()
        drawing_canvas.clear_canvas()
        drawing_canvas.commit_action()
        message = "Canvas cleared"
    elif key in COLORS:
        color_name, color = COLORS[key]
        drawing_canvas.set_color(color)
        active_tool = PEN_TOOL
        message = f"Color: {color_name}"
    elif key in (ord("+"), ord("=")):
        if active_tool == ERASER_TOOL:
            drawing_canvas.increase_eraser_size()
            message = f"Eraser size: {drawing_canvas.eraser_thickness}"
        else:
            drawing_canvas.increase_brush_size()
            message = f"Brush size: {drawing_canvas.brush_thickness}"
    elif key in (ord("-"), ord("_")):
        if active_tool == ERASER_TOOL:
            drawing_canvas.decrease_eraser_size()
            message = f"Eraser size: {drawing_canvas.eraser_thickness}"
        else:
            drawing_canvas.decrease_brush_size()
            message = f"Brush size: {drawing_canvas.brush_thickness}"
    elif key in (ord("e"), ord("E")):
        active_tool = ERASER_TOOL
        message = "Tool: ERASER"
    elif key in (ord("p"), ord("P")):
        active_tool = PEN_TOOL
        message = "Tool: PEN"
    elif key in (ord("a"), ord("A")):
        active_tool = SHAPE_TOOL
        smart_shape_enabled = True
        message = "Tool: SHAPE"
    elif key in (ord("l"), ord("L")):
        active_tool = TEXT_TOOL
        message = "Tool: TEXT"

    return active_tool, color_name, smart_shape_enabled, message


def get_status_hint(status):
    """Small product hint shown near the toolbar."""
    if status == "DRAW_LOCK":
        return "Drawing locked"

    if status == "HOLDING...":
        return "Hold steady to draw"

    if status == "SELECT/PAUSE":
        return "Two fingers to pause"

    return "Hold steady to draw"


def show_startup_screen():
    """Show a polished intro screen before the webcam loop starts."""
    screen = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    screen[:] = (12, 14, 18)

    cv2.putText(
        screen,
        "Touchless AI Drawing Studio",
        (90, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (245, 248, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        screen,
        "Draw in the air using gesture recognition",
        (92, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (185, 205, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        screen,
        "Hold index finger steady to draw | Two fingers to pause | M enhance camera",
        (92, 405),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (120, 230, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_NAME, screen)
    cv2.waitKey(2000)


def cycle_visual_mode(current_mode):
    """Move CLEAN -> AI -> DEMO -> CLEAN."""
    current_index = VISUAL_MODES.index(current_mode)
    return VISUAL_MODES[(current_index + 1) % len(VISUAL_MODES)]


def save_screenshot(frame, folder_path="assets/saved_drawings"):
    """Save the full visible frame, including camera, drawing, and UI."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = folder / f"screenshot_{timestamp}.png"
    cv2.imwrite(str(file_path), frame)
    return file_path


def save_screenshot_for_page(frame, page_number, folder_path="assets/saved_drawings"):
    """Save the full visible frame with page number in the filename."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = folder / f"screenshot_page_{page_number}_{timestamp}.png"
    cv2.imwrite(str(file_path), frame)
    return file_path


def finish_smart_stroke(drawing_canvas, shape_detector, stroke_points, shape_correction_enabled):
    """Finalize one completed stroke and optionally replace it with a clean shape."""
    if not stroke_points:
        drawing_canvas.clear_smart_stroke_backup()
        drawing_canvas.discard_action()
        return UNKNOWN

    if not shape_correction_enabled:
        drawing_canvas.clear_smart_stroke_backup()
        drawing_canvas.commit_action()
        return UNKNOWN

    shape_result = shape_detector.detect(stroke_points)
    shape_name = shape_result["name"]

    if shape_name == UNKNOWN:
        drawing_canvas.clear_smart_stroke_backup()
        drawing_canvas.commit_action()
        return UNKNOWN

    drawing_canvas.restore_smart_stroke_backup()
    drawing_canvas.draw_detected_shape(shape_result)
    drawing_canvas.clear_smart_stroke_backup()
    drawing_canvas.commit_action()
    return shape_name


def finish_diagram_stroke(drawing_canvas, shape_detector, diagram_engine, stroke_points):
    """Finalize a diagram stroke and store detected diagram objects."""
    if not stroke_points:
        drawing_canvas.clear_smart_stroke_backup()
        drawing_canvas.discard_action()
        return UNKNOWN

    shape_result = shape_detector.detect(stroke_points)
    shape_name = shape_result["name"]

    if shape_name != UNKNOWN:
        diagram_engine.add_shape_result(shape_result)

    drawing_canvas.clear_smart_stroke_backup()
    drawing_canvas.commit_action()
    return shape_name


def finish_current_stroke(
    drawing_canvas,
    shape_detector,
    diagram_engine,
    stroke_points,
    active_tool,
    smart_shape_enabled,
    diagram_mode,
):
    """Finalize a stroke using either diagram mode or normal tool mode."""
    if diagram_mode:
        return finish_diagram_stroke(
            drawing_canvas,
            shape_detector,
            diagram_engine,
            stroke_points,
        )

    return finish_smart_stroke(
        drawing_canvas,
        shape_detector,
        stroke_points,
        smart_shape_enabled and active_tool == SHAPE_TOOL,
    )


def board_background(frame_shape, board_color):
    """Create a white or dark digital board background."""
    if board_color == DARK_BOARD:
        return np.full(frame_shape, (24, 26, 30), dtype=np.uint8)

    return np.full(frame_shape, (248, 248, 245), dtype=np.uint8)


def color_from_name(color_name):
    """Return the BGR color for the current toolbar color name."""
    if color_name == "BLACK":
        return (0, 0, 0)

    for _, (name, color) in COLORS.items():
        if name == color_name:
            return color

    return (245, 245, 245)


def place_text_on_canvas(drawing_canvas, text_tool, cursor_point):
    """Place the selected text at the cursor and make it undoable."""
    if cursor_point is None:
        return False

    drawing_canvas.begin_action()
    drawing_canvas.draw_text(
        text=text_tool.current_text,
        position=cursor_point,
        font_scale=text_tool.text_size,
        color=drawing_canvas.color,
    )
    drawing_canvas.commit_action()
    return True


def main():
    try:
        tracker = HandTracker()
    except FileNotFoundError as error:
        print(error)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        tracker.close()
        return
    configure_camera(cap)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    show_startup_screen()

    previous_time = time.time()
    page_manager = None
    drawing_canvas = None
    gesture_detector = GestureDetector()
    interaction = HoldInteractionEngine(HOLD_SECONDS, HOLD_RADIUS)
    cursor_smoother = ExponentialPointSmoother(alpha=0.25, deadzone=2)
    shape_detector = ShapeDetector()
    diagram_engine = DiagramEngine()
    text_tool = TextTool()
    toolbar = UIToolbar()
    visual_overlay = VisualOverlay()

    active_tool = PEN_TOOL
    color_name = "RED"
    message = ""
    message_time = 0
    show_landmarks = False
    visual_mode = CLEAN_MODE
    latest_display_frame = None
    previous_status = GestureDetector.IDLE
    stroke_points = []
    smart_shape_enabled = False
    last_detected_shape = UNKNOWN
    enhancement_enabled = True
    app_mode = CAMERA_MODE
    board_color = WHITE_BOARD
    diagram_mode = False

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read from webcam.")
            break

        frame, _ = resize_with_aspect_ratio(frame, WINDOW_WIDTH, WINDOW_HEIGHT)
        if enhancement_enabled:
            frame = enhance_frame(frame)
        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000)

        if page_manager is None:
            page_manager = PageManager(frame.shape)
            drawing_canvas = page_manager.get_current_canvas()

        hand_result = tracker.detect(frame, timestamp_ms)
        landmarks = tracker.get_landmarks(frame)
        points = {landmark_id: (x, y) for landmark_id, x, y in landmarks}
        raw_index_tip = points.get(8)
        cursor_point = cursor_smoother.update(raw_index_tip)

        gesture_mode = gesture_detector.get_mode(landmarks, hand_result)
        is_index_only = gesture_detector.is_index_only(landmarks)
        status, hold_progress = interaction.update(
            gesture_mode=gesture_mode,
            cursor_point=cursor_point,
            is_index_only=is_index_only,
        )

        if active_tool == TEXT_TOOL:
            interaction.stop()
            status = "AIM" if cursor_point is not None else GestureDetector.IDLE
            hold_progress = 0.0

        if previous_status == "DRAW_LOCK" and status != "DRAW_LOCK":
            last_detected_shape = finish_current_stroke(
                drawing_canvas,
                shape_detector,
                diagram_engine,
                stroke_points,
                active_tool,
                smart_shape_enabled,
                diagram_mode,
            )
            stroke_points = []
            message = f"Shape: {last_detected_shape}"
            message_time = time.time()

        if status == "DRAW_LOCK" and cursor_point is not None and active_tool != TEXT_TOOL:
            if interaction.just_locked:
                stroke_points = [cursor_point]
                drawing_canvas.begin_action()
                if diagram_mode:
                    diagram_engine.ensure_base(drawing_canvas)
                drawing_canvas.begin_smart_stroke()
                drawing_canvas.start_stroke(cursor_point)
            elif active_tool == ERASER_TOOL:
                drawing_canvas.draw_eraser(cursor_point)
            else:
                stroke_points.append(cursor_point)
                drawing_canvas.draw_pen(cursor_point)
        else:
            drawing_canvas.reset_previous_point()

        if app_mode == WHITEBOARD_MODE:
            background = board_background(frame.shape, board_color)
            frame = drawing_canvas.overlay_on_frame(background)
        else:
            frame = drawing_canvas.overlay_on_frame(frame)

        if show_landmarks or visual_mode == AI_MODE:
            visual_overlay.draw_technical_landmarks(frame, landmarks)

        if visual_mode == DEMO_MODE:
            visual_overlay.draw_demo_overlay(frame, landmarks, status, cursor_point)

        draw_cursor(frame, cursor_point, drawing_canvas.color, status, hold_progress)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        toolbar.draw(
            frame=frame,
            selected_tool=active_tool,
            selected_color_name=color_name,
            selected_color=drawing_canvas.color,
            brush_size=drawing_canvas.brush_thickness,
            fps=fps,
            mode=status,
            status_text=get_status_hint(status),
            visual_mode=visual_mode,
            smart_shape_enabled=smart_shape_enabled,
            last_detected_shape=last_detected_shape,
            enhancement_enabled=enhancement_enabled,
            app_mode=app_mode,
            board_color=board_color,
            page_label=page_manager.page_label(),
            text_preview=text_tool.current_text,
            text_size=text_tool.text_size,
            diagram_mode=diagram_mode,
            diagram_object_count=diagram_engine.object_count(),
            message=message,
            message_active=bool(message) and time.time() - message_time <= 2,
        )

        latest_display_frame = frame.copy()
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKeyEx(1)

        if key in (ord("q"), ord("Q")):
            if status == "DRAW_LOCK":
                finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
            break

        if key == 32:
            if active_tool == TEXT_TOOL:
                if place_text_on_canvas(drawing_canvas, text_tool, cursor_point):
                    message = f"Text placed: {text_tool.current_text}"
                else:
                    message = "Move cursor to place text"
                message_time = time.time()
                previous_status = GestureDetector.IDLE
                continue

            if status == "DRAW_LOCK":
                last_detected_shape = finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
                stroke_points = []
            interaction.stop()
            drawing_canvas.reset_previous_point()
            message = "Drawing stopped"
            message_time = time.time()
            previous_status = GestureDetector.IDLE
            continue

        if key in (13, 10):
            if active_tool == TEXT_TOOL:
                if place_text_on_canvas(drawing_canvas, text_tool, cursor_point):
                    message = f"Text placed: {text_tool.current_text}"
                else:
                    message = "Move cursor to place text"
                message_time = time.time()
            elif diagram_mode:
                if diagram_engine.layout_and_draw(drawing_canvas):
                    message = "Diagram layout applied"
                else:
                    message = "No diagram objects"
                message_time = time.time()
            continue

        if key in (ord("w"), ord("W")):
            app_mode = WHITEBOARD_MODE if app_mode == CAMERA_MODE else CAMERA_MODE
            if app_mode == WHITEBOARD_MODE and board_color == WHITE_BOARD:
                drawing_canvas.set_color((0, 0, 0))
                color_name = "BLACK"
            message = f"Mode: {app_mode}"
            message_time = time.time()
            continue

        if key in (ord("b"), ord("B")):
            board_color = DARK_BOARD if board_color == WHITE_BOARD else WHITE_BOARD
            if app_mode == WHITEBOARD_MODE and board_color == WHITE_BOARD:
                drawing_canvas.set_color((0, 0, 0))
                color_name = "BLACK"
            elif app_mode == WHITEBOARD_MODE and board_color == DARK_BOARD and color_name == "BLACK":
                drawing_canvas.set_color((245, 245, 245))
                color_name = "WHITE"
            message = f"Board: {board_color}"
            message_time = time.time()
            continue

        if key in (ord("n"), ord("N")):
            if status == "DRAW_LOCK":
                finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
                stroke_points = []
            drawing_canvas = page_manager.new_page()
            diagram_engine.reset()
            drawing_canvas.set_color(color_from_name(color_name))
            interaction.stop()
            cursor_smoother.reset()
            stroke_points = []
            message = f"New page {page_manager.page_label()}"
            message_time = time.time()
            continue

        if key in (ord("["), ord("{")):
            text_tool.decrease_size()
            message = f"Text size: {text_tool.text_size:.1f}"
            message_time = time.time()
            continue

        if key in (ord("]"), ord("}")):
            text_tool.increase_size()
            message = f"Text size: {text_tool.text_size:.1f}"
            message_time = time.time()
            continue

        if key == 2424832:
            if status == "DRAW_LOCK":
                finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
                stroke_points = []
            drawing_canvas = page_manager.previous_page()
            diagram_engine.reset()
            interaction.stop()
            cursor_smoother.reset()
            stroke_points = []
            message = f"Page {page_manager.page_label()}"
            message_time = time.time()
            continue

        if key == 2555904:
            if status == "DRAW_LOCK":
                finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
                stroke_points = []
            drawing_canvas = page_manager.next_page()
            diagram_engine.reset()
            interaction.stop()
            cursor_smoother.reset()
            stroke_points = []
            message = f"Page {page_manager.page_label()}"
            message_time = time.time()
            continue

        if key in (ord("z"), ord("Z"), 26):
            message = "Undo" if drawing_canvas.undo() else "Nothing to undo"
            message_time = time.time()
            continue

        if key in (ord("y"), ord("Y"), 25):
            message = "Redo" if drawing_canvas.redo() else "Nothing to redo"
            message_time = time.time()
            continue

        if key in (ord("v"), ord("V")):
            visual_mode = cycle_visual_mode(visual_mode)
            message = f"Visual mode: {visual_mode}"
            message_time = time.time()
            continue

        if key in (ord("h"), ord("H")):
            show_landmarks = not show_landmarks
            message = "Technical landmarks on" if show_landmarks else "Technical landmarks off"
            message_time = time.time()
            continue

        if key in (ord("t"), ord("T")):
            if status == "DRAW_LOCK":
                last_detected_shape = finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
                stroke_points = []
                interaction.stop()
                drawing_canvas.reset_previous_point()
            smart_shape_enabled = not smart_shape_enabled
            message = f"Smart Shape: {'ON' if smart_shape_enabled else 'OFF'}"
            message_time = time.time()
            continue

        if key in (ord("d"), ord("D")):
            if status == "DRAW_LOCK":
                last_detected_shape = finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
                stroke_points = []
                interaction.stop()
                drawing_canvas.reset_previous_point()
            diagram_mode = not diagram_mode
            if diagram_mode:
                diagram_engine.reset()
                diagram_engine.ensure_base(drawing_canvas)
                active_tool = SHAPE_TOOL
                smart_shape_enabled = True
            else:
                diagram_engine.reset()
            message = f"Diagram Mode: {'ON' if diagram_mode else 'OFF'}"
            message_time = time.time()
            continue

        if key in (ord("l"), ord("L")):
            if status == "DRAW_LOCK":
                finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
                stroke_points = []
                interaction.stop()
                drawing_canvas.reset_previous_point()
            active_tool = TEXT_TOOL
            text_tool.prompt_for_text()
            message = f"Text: {text_tool.current_text}"
            message_time = time.time()
            continue

        if key in TextTool.QUICK_NOTES:
            text_tool.set_quick_note_from_key(key)
            active_tool = TEXT_TOOL
            message = f"Text: {text_tool.current_text}"
            message_time = time.time()
            continue

        if key in (ord("m"), ord("M")):
            enhancement_enabled = not enhancement_enabled
            message = f"Enhancement: {'ON' if enhancement_enabled else 'OFF'}"
            message_time = time.time()
            continue

        if key in (ord("S"), ord("x"), ord("X")):
            if latest_display_frame is not None:
                screenshot_path = save_screenshot_for_page(
                    latest_display_frame,
                    page_manager.page_number(),
                )
                message = f"Screenshot: {screenshot_path.name}"
                message_time = time.time()
            continue

        if key == ord("s"):
            save_background = (24, 26, 30) if app_mode == WHITEBOARD_MODE and board_color == DARK_BOARD else (255, 255, 255)
            saved_path = drawing_canvas.save_canvas_for_page(
                page_manager.page_number(),
                background_color=save_background,
            )
            message = f"Saved: {saved_path.name}"
            message_time = time.time()
            continue

        if key != -1:
            if status == "DRAW_LOCK":
                last_detected_shape = finish_current_stroke(
                    drawing_canvas,
                    shape_detector,
                    diagram_engine,
                    stroke_points,
                    active_tool,
                    smart_shape_enabled,
                    diagram_mode,
                )
                stroke_points = []
                interaction.stop()
                drawing_canvas.reset_previous_point()

            active_tool, color_name, smart_shape_enabled, new_message = handle_key(
                key,
                drawing_canvas,
                active_tool,
                color_name,
                smart_shape_enabled,
            )
            if new_message:
                interaction.stop()
                drawing_canvas.reset_previous_point()
                message = new_message
                message_time = time.time()

        previous_status = status

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
