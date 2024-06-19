# Standard library imports
from typing import TypeAlias, TYPE_CHECKING

# Third-party imports
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np

# Local library imports
from calibrationpinn.inputoutput import PathAdministrator
from calibrationpinn.simulations.plotters.domain2DPlotter import Domain2DPlotter
from calibrationpinn.simulations.plotters.domain2DWithHolePlotterConfig import (
    Domain2DWithHolePlotterConfig,
)
from calibrationpinn.simulations.simulationResults import SimulationResults2D
from calibrationpinn.typeAliases import NPArray, NPFloat, PLTAxes, PLTFigure


PlotterConfig: TypeAlias = Domain2DWithHolePlotterConfig
SimulationResults: TypeAlias = SimulationResults2D


class Domain2DWithHolePlotter(Domain2DPlotter):
    def __init__(self, output_subdir_name: str, path_admin: PathAdministrator) -> None:
        super().__init__(output_subdir_name, path_admin)

    def _preprocess_simulation_results(
        self,
        simulation_results: SimulationResults,
        plotter_config: PlotterConfig,
    ):
        x_coordinates = simulation_results.coordinates_grid_x
        y_coordinates = simulation_results.coordinates_grid_y
        radius = plotter_config.radius_hole
        num_points_per_dim = simulation_results.coordinates_grid_x.shape[0]
        length_dim = np.nanmax(simulation_results.coordinates_grid_x) - np.nanmin(
            simulation_results.coordinates_grid_x
        )
        length_per_point = length_dim / num_points_per_dim
        substitution_hole = 0.0
        mask = np.sqrt(x_coordinates**2 + y_coordinates**2) >= (
            radius - length_per_point
        )
        return SimulationResults2D(
            coordinates_grid_x=np.where(
                mask, simulation_results.coordinates_grid_x, substitution_hole
            ),
            coordinates_grid_y=np.where(
                mask, simulation_results.coordinates_grid_y, substitution_hole
            ),
            prediction_grid=np.where(
                mask, simulation_results.prediction_grid, substitution_hole
            ),
            solution_grid=np.where(
                mask, simulation_results.solution_grid, substitution_hole
            ),
            residual_grid=np.where(
                mask, simulation_results.residual_grid, substitution_hole
            ),
            relative_error_grid=np.where(
                mask, simulation_results.relative_error_grid, substitution_hole
            ),
        )

    def _plot_one_result_grid(
        self,
        result_grid: NPArray,
        simulation_results: SimulationResults,
        normalizer: BoundaryNorm,
        cbar_ticks: list[NPFloat],
        figure: PLTFigure,
        axes: PLTAxes,
        plotter_config: PlotterConfig,
    ) -> None:
        grid_cut = self._cut_result_grid_for_plot(result_grid)
        plot = axes.pcolormesh(
            simulation_results.coordinates_grid_x,
            simulation_results.coordinates_grid_y,
            grid_cut,
            cmap=plt.get_cmap(plotter_config.color_map),
            norm=normalizer,
        )
        cbar = figure.colorbar(mappable=plot, ax=axes, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(map(str, cbar_ticks))
        cbar.ax.minorticks_off()
        figure.axes[1].tick_params(labelsize=plotter_config.font_size)
        self._add_hole(axes, plotter_config)

    def _add_hole(self, axes, plotter_config):
        hole = plt.Circle(
            (plotter_config.hole_center_x, plotter_config.hole_center_y),
            radius=plotter_config.radius_hole,
            color="white",
        )
        axes.add_patch(hole)
