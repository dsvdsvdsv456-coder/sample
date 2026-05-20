class ExponentialPointSmoother:
    """Smooths noisy fingertip points with an exponential moving average."""

    def __init__(self, alpha=0.25, deadzone=2):
        self.alpha = alpha
        self.deadzone = deadzone
        self.smoothed_point = None

    def update(self, point):
        """Return a smoother version of the current point."""
        if point is None:
            self.reset()
            return None

        if self.smoothed_point is None:
            self.smoothed_point = point
            return point

        if self._distance(self.smoothed_point, point) <= self.deadzone:
            return self.smoothed_point

        current_x, current_y = point
        previous_x, previous_y = self.smoothed_point

        smoothed_x = self.alpha * current_x + (1 - self.alpha) * previous_x
        smoothed_y = self.alpha * current_y + (1 - self.alpha) * previous_y

        self.smoothed_point = (int(smoothed_x), int(smoothed_y))
        return self.smoothed_point

    def reset(self):
        """Forget the previous smoothed point."""
        self.smoothed_point = None

    def _distance(self, first_point, second_point):
        first_x, first_y = first_point
        second_x, second_y = second_point
        return ((second_x - first_x) ** 2 + (second_y - first_y) ** 2) ** 0.5
