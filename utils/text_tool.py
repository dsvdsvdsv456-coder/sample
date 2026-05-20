class TextTool:
    """Stores the selected text and text size for TEXT mode."""

    QUICK_NOTES = {
        ord("6"): "Important",
        ord("7"): "Remember",
        ord("8"): "Example",
        ord("9"): "Definition",
    }

    def __init__(self, default_text="Note", default_size=1.0):
        self.current_text = default_text
        self.text_size = default_size

    def prompt_for_text(self):
        """Ask for text in the terminal and keep the previous text if empty."""
        user_text = input("Enter text: ").strip()
        if user_text:
            self.current_text = user_text
        return self.current_text

    def set_quick_note_from_key(self, key):
        """Select a preset note label from shortcut keys 6-9."""
        if key in self.QUICK_NOTES:
            self.current_text = self.QUICK_NOTES[key]
            return True
        return False

    def increase_size(self, amount=0.1, max_size=3.0):
        """Make text larger."""
        self.text_size = min(max_size, round(self.text_size + amount, 2))

    def decrease_size(self, amount=0.1, min_size=0.4):
        """Make text smaller."""
        self.text_size = max(min_size, round(self.text_size - amount, 2))
