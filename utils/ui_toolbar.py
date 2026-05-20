import cv2


class UIToolbar:
    """Minimal, lightweight UI overlay for the drawing app."""

    def __init__(self):
        self.toolbar_height = 50

        self.colors = [
            ("RED", (38, 54, 235)),
            ("GREEN", (73, 190, 92)),
            ("BLUE", (238, 132, 67)),
            ("YELLOW", (42, 210, 245)),
            ("WHITE", (245, 245, 245)),
        ]

    def draw(
        self,
        frame,
        selected_tool,
        selected_color_name,
        selected_color,
        brush_size,
        fps,
        mode,
        status_text,
        visual_mode,
        smart_shape_enabled,
        last_detected_shape,
        enhancement_enabled,
        app_mode,
        board_color,
        page_label,
        text_preview,
        text_size,
        diagram_mode,
        diagram_object_count,
        message="",
        message_active=False,
    ):
        """Draw the complete minimal UI overlay."""
        self._draw_top_toolbar(frame, selected_tool, selected_color_name, brush_size)
        self._draw_info_text(
            frame,
            fps,
            mode,
            selected_tool,
            visual_mode,
            smart_shape_enabled,
            last_detected_shape,
            enhancement_enabled,
            app_mode,
            board_color,
            page_label,
            text_preview,
            text_size,
            diagram_mode,
            diagram_object_count,
        )
        self._draw_status_pill(frame, status_text)
        self._draw_help_text(frame)

        if message_active:
            self._draw_message(frame, message)

        return frame

    def _draw_top_toolbar(self, frame, selected_tool, selected_color_name, brush_size):
        """Draw a thin semi-transparent toolbar at the top."""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, self.toolbar_height), (18, 20, 24), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        x = 16
        x = self._draw_text_button(frame, "PEN", x, selected_tool == "PEN")
        x = self._draw_text_button(frame, "ERASER", x, selected_tool == "ERASER")
        x = self._draw_text_button(frame, "SHAPE", x, selected_tool == "SHAPE")
        x = self._draw_text_button(frame, "TEXT", x, selected_tool == "TEXT")

        for color_name, color in self.colors:
            selected = color_name == selected_color_name
            self._draw_color_dot(frame, x + 14, 25, color, selected)
            x += 34

        x += 8
        x = self._draw_text_button(frame, "CLEAR", x, False)
        x = self._draw_text_button(frame, "SAVE", x, False)

        self._draw_text(
            frame,
            f"Brush {brush_size}",
            (width - 112, 31),
            scale=0.5,
            color=(235, 238, 242),
            thickness=1,
        )

    def _draw_text_button(self, frame, label, x, selected):
        """Draw a small text-only toolbar button."""
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        button_width = text_width + 22
        color = (245, 248, 252) if selected else (195, 202, 210)

        if selected:
            cv2.line(frame, (x + 8, 41), (x + button_width - 8, 41), (85, 197, 255), 2, cv2.LINE_AA)

        self._draw_text(frame, label, (x + 10, 30), scale=0.52, color=color, thickness=1)
        return x + button_width + 8

    def _draw_color_dot(self, frame, x, y, color, selected):
        """Draw one small color control dot."""
        radius = 8
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)

        if selected:
            cv2.circle(frame, (x, y), radius + 5, (85, 197, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), radius + 1, (215, 220, 225), 1, cv2.LINE_AA)

    def _draw_info_text(
        self,
        frame,
        fps,
        mode,
        selected_tool,
        visual_mode,
        smart_shape_enabled,
        last_detected_shape,
        enhancement_enabled,
        app_mode,
        board_color,
        page_label,
        text_preview,
        text_size,
        diagram_mode,
        diagram_object_count,
    ):
        """Draw minimal status text in the top-left below the toolbar."""
        x, y = 16, self.toolbar_height + 24
        color = (240, 242, 245)
        smart_text = "ON" if smart_shape_enabled else "OFF"
        enhance_text = "ON" if enhancement_enabled else "OFF"

        self._draw_text(frame, f"FPS  {fps:.1f}", (x, y), 0.5, color, 1)
        self._draw_text(frame, f"Gesture {mode}", (x, y + 23), 0.5, color, 1)
        self._draw_text(frame, f"Tool {selected_tool}", (x, y + 46), 0.5, color, 1)
        self._draw_text(frame, f"View {visual_mode}", (x, y + 69), 0.5, color, 1)
        self._draw_text(frame, f"Smart Shape {smart_text}", (x, y + 92), 0.5, color, 1)
        self._draw_text(frame, f"Last Shape {last_detected_shape}", (x, y + 115), 0.5, color, 1)
        self._draw_text(frame, f"Enhancement {enhance_text}", (x, y + 138), 0.5, color, 1)
        self._draw_text(frame, f"Mode {app_mode}", (x, y + 161), 0.5, color, 1)
        self._draw_text(frame, f"Board {board_color}", (x, y + 184), 0.5, color, 1)
        self._draw_text(frame, f"Page {page_label}", (x, y + 207), 0.5, color, 1)
        preview = text_preview[:18] + "..." if len(text_preview) > 18 else text_preview
        self._draw_text(frame, f"Text {preview}", (x, y + 230), 0.5, color, 1)
        self._draw_text(frame, f"Text Size {text_size:.1f}", (x, y + 253), 0.5, color, 1)
        diagram_text = "ON" if diagram_mode else "OFF"
        self._draw_text(frame, f"Diagram Mode {diagram_text}", (x, y + 276), 0.5, color, 1)
        self._draw_text(frame, f"Objects {diagram_object_count}", (x, y + 299), 0.5, color, 1)

    def _draw_help_text(self, frame):
        """Draw a single thin help line at the bottom."""
        height = frame.shape[0]
        text = "D Diagram | Enter Layout | L Text | 6-9 Notes | W Board | N New | Z Undo | Y Redo | S Save | Q Quit"
        self._draw_text(frame, text, (16, height - 18), 0.5, (235, 238, 242), 1)

    def _draw_status_pill(self, frame, status_text):
        """Draw a tiny status hint without blocking the camera."""
        if not status_text:
            return

        width = frame.shape[1]
        text_width = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        x = width - text_width - 34
        y = self.toolbar_height + 28

        self._draw_text(frame, status_text, (x, y), 0.52, (230, 245, 255), 1)

    def _draw_message(self, frame, message):
        """Draw a subtle temporary message near the top center."""
        if not message:
            return

        width = frame.shape[1]
        text_width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        x = max(16, (width - text_width) // 2)
        self._draw_text(frame, message, (x, self.toolbar_height + 25), 0.55, (120, 245, 180), 1)

    def _draw_text(self, frame, text, position, scale, color, thickness):
        """Draw small anti-aliased text."""
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
