# Standard library imports

# Third library imports

# Local library imports
from calibrationpinn.configs import ABCPlotterConfig


class TrainingLoggerPlotterConfig(ABCPlotterConfig):
    def __init__(self, title: str, save_title: str, x_label: str, y_label: str) -> None:
        super().__init__()
        self.title = title
        self.save_title = save_title
        self.x_label = x_label
        self.y_label = y_label
