from utils.drawing_canvas import DrawingCanvas


class PageManager:
    """Manages multiple independent whiteboard pages."""

    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.pages = [DrawingCanvas(frame_shape)]
        self.current_index = 0

    def get_current_canvas(self):
        """Return the DrawingCanvas for the active page."""
        return self.pages[self.current_index]

    def new_page(self):
        """Create a new blank page after the current page and switch to it."""
        new_canvas = DrawingCanvas(self.frame_shape)
        insert_index = self.current_index + 1
        self.pages.insert(insert_index, new_canvas)
        self.current_index = insert_index
        return self.get_current_canvas()

    def next_page(self):
        """Move to the next page when one exists."""
        if self.current_index < len(self.pages) - 1:
            self.current_index += 1
        return self.get_current_canvas()

    def previous_page(self):
        """Move to the previous page when one exists."""
        if self.current_index > 0:
            self.current_index -= 1
        return self.get_current_canvas()

    def page_number(self):
        """Return the human-friendly page number."""
        return self.current_index + 1

    def page_count(self):
        """Return the total number of pages."""
        return len(self.pages)

    def page_label(self):
        """Return text like 1/3 for the UI."""
        return f"{self.page_number()}/{self.page_count()}"
