# Standard library imports

# Third-party imports

# Local library imports
from calibrationpinn.configs import ABCPlotterConfig


class Domain2DPlotterConfig(ABCPlotterConfig):
    def __init__(self, simulation_object: str, save_title_identifier: str) -> None:
        super().__init__()
        self.simulation_object = simulation_object
        self.save_title_identifier = save_title_identifier

        self.title_pad = 10

        self.title_prefix_prediction = "Approximation"
        self.title_prefix_solution = "Solution"
        self.title_prefix_residual = "Absolute error"
        self.title_prefix_relative_error = "Relative error"

        self.save_title_suffix_prediction = "prediction"
        self.save_title_suffix_solution = "solution"
        self.save_title_suffix_residual = "absolute_error"
        self.save_title_suffix_relative_error = "relative_error"

        self.x_label = "x [mm]"
        self.y_label = "y [mm]"

        self.color_map = "jet"
        self.ticks_max_number_of_intervals = 255
        self.num_cbar_ticks = 7
