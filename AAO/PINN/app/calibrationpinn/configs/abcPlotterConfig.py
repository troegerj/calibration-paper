# Standard library imports

# Third-party imports

# Local library imports


class ABCPlotterConfig:
    def __init__(self) -> None:
        # font sizes
        self.label_size = 24
        # font size in legend
        self.font_size = 20
        self.font = {"size": self.label_size}

        # major ticks
        self.major_tick_label_size = 24
        self.major_ticks_size = self.font_size
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 14
        self.minor_ticks_size = 12
        self.minor_ticks_width = 1

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save format
        self.dpi = 300
        self.save_format = "pdf"
