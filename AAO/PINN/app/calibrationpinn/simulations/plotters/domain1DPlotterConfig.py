# Standard library imports

# Third-party imports

# Local library imports
from calibrationpinn.configs import ABCPlotterConfig


class Domain1DPlotterConfig(ABCPlotterConfig):
    def __init__(
        self,
        simulation_object: str,
        save_title_identifier: str,
        x_label: str,
        y_label: str,
    ) -> None:
        super().__init__()
        self.simulation_object = simulation_object
        self.save_title_identifier = save_title_identifier
        self.x_label = x_label
        self.y_label = y_label

        self.title_pad = 10

        self.title_prefix_solution_and_prediction = "Prediction"
        self.title_prefix_residual = "Absolute error"
        self.title_prefix_relative_error = "Relative error"

        self.save_title_suffix_solution_and_prediction = "prediction"
        self.save_title_suffix_residual = "absolute_error"
        self.save_title_suffix_relative_error = "relative_error"

        # line or marker style
        self.format_string_solution = "o"
        self.format_string_prediction = "-"
        self.format_string_errors = "-"

        # legend
        self.label_solution = "data"
        self.label_prediction = "PINN"
        self.label_residual = "residual"
        self.label_relative_error = "rel. error"
