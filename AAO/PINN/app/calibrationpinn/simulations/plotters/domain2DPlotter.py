# Standard library imports
from typing import Optional, TypeAlias, TYPE_CHECKING

# Third-party imports
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Local library imports
from calibrationpinn.inputoutput import PathAdministrator
from calibrationpinn.simulations.plotters.domain2DPlotterConfig import (
    Domain2DPlotterConfig,
)
from calibrationpinn.simulations.simulationResults import SimulationResults2D
from calibrationpinn.typeAliases import PLTFigure, PLTAxes, NPArray, NPFloat


PlotterConfig: TypeAlias = Domain2DPlotterConfig
SimulationResults: TypeAlias = SimulationResults2D


class Domain2DPlotter:
    def __init__(self, output_subdir_name: str, path_admin: PathAdministrator) -> None:
        self._output_subdir_name = output_subdir_name
        self._path_admin = path_admin
        self._fig_prediction: Optional[PLTFigure] = None
        self._axes_prediction: Optional[PLTAxes] = None
        self._fig_solution: Optional[PLTFigure] = None
        self._axes_solution: Optional[PLTAxes] = None
        self._fig_residual: Optional[PLTFigure] = None
        self._axes_residual: Optional[PLTAxes] = None
        self._fig_relative_error: Optional[PLTFigure] = None
        self._axes_relative_error: Optional[PLTAxes] = None
        self._normalizer_prediction_and_solution: Optional[BoundaryNorm] = None
        self._normalizer_residual: Optional[BoundaryNorm] = None
        self._normalizer_relative_error: Optional[BoundaryNorm] = None
        self._ticks_prediction_and_solution: Optional[list[NPFloat]] = None
        self._ticks_residual: Optional[list[NPFloat]] = None
        self._ticks_relative_error: Optional[list[NPFloat]] = None

    def plot(
        self,
        simulation_results: SimulationResults,
        plotter_config: PlotterConfig,
    ) -> None:
        self._reset()
        simulation_results = self._preprocess_simulation_results(
            simulation_results, plotter_config
        )
        self._set_up(simulation_results, plotter_config)
        self._plot(simulation_results, plotter_config)
        self._save_plots(plotter_config)
        plt.close("all")

    def _reset(self) -> None:
        self._fig_prediction = None
        self._axes_prediction = None
        self._fig_solution = None
        self._axes_solution = None
        self._fig_residual = None
        self._axes_residual = None
        self._fig_relative_error = None
        self._axes_relative_error = None
        self._normalizer_prediction_and_solution = None
        self._normalizer_residual = None
        self._normalizer_relative_error = None

    def _preprocess_simulation_results(
        self,
        simulation_results: SimulationResults,
        plotter_config: PlotterConfig,
    ):
        return simulation_results

    def _set_up(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ):
        self._set_up_figures_and_axes(simulation_results, plotter_config)
        self._set_up_titles(plotter_config)
        self._set_up_normalizers(simulation_results, plotter_config)
        self._set_up_ticks(simulation_results, plotter_config)

    def _set_up_figures_and_axes(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        self._fig_prediction, self._axes_prediction = self._set_up_one_figure_and_axes(
            simulation_results, plotter_config
        )
        self._fig_solution, self._axes_solution = self._set_up_one_figure_and_axes(
            simulation_results, plotter_config
        )
        self._fig_residual, self._axes_residual = self._set_up_one_figure_and_axes(
            simulation_results, plotter_config
        )
        (
            self._fig_relative_error,
            self._axes_relative_error,
        ) = self._set_up_one_figure_and_axes(simulation_results, plotter_config)

    def _set_up_one_figure_and_axes(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> tuple[PLTFigure, PLTAxes]:
        figure, axes = plt.subplots()
        self._set_up_axis_labels(axes, plotter_config)
        self._set_up_axis_limits(axes, simulation_results)
        return figure, axes

    def _set_up_axis_labels(self, axes: PLTAxes, plotter_config: PlotterConfig) -> None:
        axes.set_xlabel(plotter_config.x_label, **plotter_config.font)
        axes.set_ylabel(plotter_config.y_label, **plotter_config.font)
        axes.tick_params(
            axis="both", which="minor", labelsize=plotter_config.minor_tick_label_size
        )
        axes.tick_params(
            axis="both", which="major", labelsize=plotter_config.major_tick_label_size
        )

    def _set_up_axis_limits(
        self, axes: PLTAxes, simulation_results: SimulationResults
    ) -> None:
        coordinates_grid_x = simulation_results.coordinates_grid_x
        coordinates_grid_y = simulation_results.coordinates_grid_y
        x_min = np.nanmin(coordinates_grid_x)
        x_max = np.nanmax(coordinates_grid_x)
        y_min = np.nanmin(coordinates_grid_y)
        y_max = np.nanmax(coordinates_grid_y)
        x_ticks = np.linspace(x_min, x_max, num=3, endpoint=True)
        y_ticks = np.linspace(y_min, y_max, num=3, endpoint=True)
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(map(str, x_ticks))
        axes.set_yticks(y_ticks)
        axes.set_yticklabels(map(str, y_ticks))
        axes.tick_params(axis="both", which="major", pad=15)

    def _set_up_titles(self, plotter_config: PlotterConfig) -> None:
        self._set_up_title_for_one_plot(
            self._axes_prediction,
            plotter_config.title_prefix_prediction,
            plotter_config.simulation_object,
            plotter_config,
        )
        self._set_up_title_for_one_plot(
            self._axes_solution,
            plotter_config.title_prefix_solution,
            plotter_config.simulation_object,
            plotter_config,
        )
        self._set_up_title_for_one_plot(
            self._axes_residual,
            plotter_config.title_prefix_residual,
            plotter_config.simulation_object,
            plotter_config,
        )
        self._set_up_title_for_one_plot(
            self._axes_relative_error,
            plotter_config.title_prefix_relative_error,
            plotter_config.simulation_object,
            plotter_config,
        )

    def _set_up_title_for_one_plot(
        self,
        axes: PLTAxes,
        title_prefix: str,
        simulation_object: str,
        plotter_config: PlotterConfig,
    ) -> None:
        # title = self._combine_title(title_prefix, simulation_object)
        axes.set_title(
            title_prefix, pad=plotter_config.title_pad, **plotter_config.font
        )

    def _combine_title(self, title_prefix: str, simulation_object: str) -> str:
        return title_prefix + " " + simulation_object.lower()

    def _set_up_normalizers(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        self._set_up_normalizer_for_prediction_and_solution(
            simulation_results, plotter_config
        )
        self._set_up_normalizer_for_residual(simulation_results, plotter_config)
        self._set_up_normalizer_for_relative_error(simulation_results, plotter_config)

    def _set_up_normalizer_for_prediction_and_solution(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        relevant_result_grids = self._compose_solution_and_prediction_grids(
            simulation_results
        )
        self._normalizer_prediction_and_solution = self._create_normalizer(
            plotter_config, relevant_result_grids
        )

    def _set_up_normalizer_for_residual(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        relevant_result_grids = (simulation_results.residual_grid,)
        self._normalizer_residual = self._create_normalizer(
            plotter_config, relevant_result_grids
        )

    def _set_up_normalizer_for_relative_error(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        relevant_result_grids = (simulation_results.relative_error_grid,)
        self._normalizer_relative_error = self._create_normalizer(
            plotter_config, relevant_result_grids
        )

    def _create_normalizer(
        self, plotter_config: PlotterConfig, result_grids: tuple[NPArray, ...]
    ) -> BoundaryNorm:
        stacked_result_grids = self._stack_result_grids(result_grids)
        min_value = self._determine_minimum_value(stacked_result_grids)
        max_value = self._determine_maximum_value(stacked_result_grids)
        tick_values = MaxNLocator(
            nbins=plotter_config.ticks_max_number_of_intervals
        ).tick_values(min_value, max_value)
        return BoundaryNorm(
            tick_values, ncolors=plt.get_cmap(plotter_config.color_map).N, clip=True
        )

    def _set_up_ticks(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        self._set_up_ticks_for_prediction_and_solution(
            simulation_results, plotter_config
        )
        self._set_up_ticks_for_residual(simulation_results, plotter_config)
        self._set_up_ticks_for_relative_error(simulation_results, plotter_config)

    def _set_up_ticks_for_prediction_and_solution(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        relevant_result_grids = self._compose_solution_and_prediction_grids(
            simulation_results
        )
        self._ticks_prediction_and_solution = self._create_ticks(
            plotter_config, relevant_result_grids
        )

    def _set_up_ticks_for_residual(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        relevant_result_grids = (simulation_results.residual_grid,)
        self._ticks_residual = self._create_ticks(plotter_config, relevant_result_grids)

    def _set_up_ticks_for_relative_error(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        relevant_result_grids = (simulation_results.relative_error_grid,)
        self._ticks_relative_error = self._create_ticks(
            plotter_config, relevant_result_grids
        )

    def _create_ticks(
        self, plotter_config: PlotterConfig, result_grids: tuple[NPArray, ...]
    ) -> list[NPFloat]:
        stacked_result_grids = self._stack_result_grids(result_grids)
        min_value = self._determine_minimum_value(stacked_result_grids)
        max_value = self._determine_maximum_value(stacked_result_grids)
        ticks = (
            np.linspace(
                min_value, max_value, num=plotter_config.num_cbar_ticks, endpoint=True
            )
            .round(decimals=4)
            .tolist()
        )
        return ticks

    def _compose_solution_and_prediction_grids(
        self, simulation_results: SimulationResults
    ) -> tuple[NPArray, NPArray]:
        return simulation_results.solution_grid, simulation_results.prediction_grid

    def _stack_result_grids(self, result_grids: tuple[NPArray, ...]) -> NPArray:
        return np.array(result_grids)

    def _determine_minimum_value(self, stacked_result_grids: NPArray) -> NPFloat:
        return np.nanmin(stacked_result_grids[np.isfinite(stacked_result_grids)])

    def _determine_maximum_value(self, stacked_result_grids: NPArray) -> NPFloat:
        return np.nanmax(stacked_result_grids[np.isfinite(stacked_result_grids)])

    def _plot(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        self._plot_one_result_grid(
            simulation_results.prediction_grid,
            simulation_results,
            self._normalizer_prediction_and_solution,
            self._ticks_prediction_and_solution,
            self._fig_prediction,
            self._axes_prediction,
            plotter_config,
        )
        self._plot_one_result_grid(
            simulation_results.solution_grid,
            simulation_results,
            self._normalizer_prediction_and_solution,
            self._ticks_prediction_and_solution,
            self._fig_solution,
            self._axes_solution,
            plotter_config,
        )
        self._plot_one_result_grid(
            simulation_results.residual_grid,
            simulation_results,
            self._normalizer_residual,
            self._ticks_residual,
            self._fig_residual,
            self._axes_residual,
            plotter_config,
        )
        self._plot_one_result_grid(
            simulation_results.relative_error_grid,
            simulation_results,
            self._normalizer_relative_error,
            self._ticks_relative_error,
            self._fig_relative_error,
            self._axes_relative_error,
            plotter_config,
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

    def _cut_result_grid_for_plot(self, result_grid: NPArray) -> NPArray:
        return result_grid[:-1, :-1]

    def _save_plots(self, plotter_config: PlotterConfig) -> None:
        save_format = plotter_config.save_format
        self._save_one_plot(
            self._fig_prediction,
            plotter_config.save_title_suffix_prediction,
            save_format,
            plotter_config,
        )
        self._save_one_plot(
            self._fig_solution,
            plotter_config.save_title_suffix_solution,
            save_format,
            plotter_config,
        )
        self._save_one_plot(
            self._fig_residual,
            plotter_config.save_title_suffix_residual,
            save_format,
            plotter_config,
        )
        self._save_one_plot(
            self._fig_relative_error,
            plotter_config.save_title_suffix_relative_error,
            save_format,
            plotter_config,
        )

    def _save_one_plot(
        self,
        figure: PLTFigure,
        plot_type: str,
        save_format: str,
        plotter_config: PlotterConfig,
    ) -> None:
        file_name = self._creat_file_name(
            plotter_config.save_title_identifier,
            plot_type,
            save_format,
        )
        save_path = self._path_admin.get_path_to_output_file(
            file_name, self._output_subdir_name
        )
        figure.savefig(
            save_path, format=save_format, bbox_inches="tight", dpi=plotter_config.dpi
        )

    def _creat_file_name(
        self, simulation_object: str, plot_type: str, save_format: str
    ) -> str:
        return "plot_{simulation_object}_{plot_type}.{save_format}".format(
            simulation_object=simulation_object,
            plot_type=plot_type,
            save_format=save_format,
        )
