# Standard library imports

# Third-party imports

# Local library imports
from calibrationpinn.simulations.plotters.domain2DPlotterConfig import (
    Domain2DPlotterConfig,
)


class Domain2DWithHolePlotterConfig(Domain2DPlotterConfig):
    def __init__(
        self, simulation_object: str, save_title_identifier: str, radius_hole: float
    ) -> None:
        super().__init__(simulation_object, save_title_identifier)
        self.radius_hole = radius_hole

        self.hole_center_x = 0
        self.hole_center_y = 0
