# Standard library imports
from typing import Optional, TypeAlias, TYPE_CHECKING

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local library imports
from calibrationpinn.inputoutput import PathAdministrator
from calibrationpinn.simulations.plotters.domain1DPlotterConfig import (
    Domain1DPlotterConfig,
)
from calibrationpinn.simulations.simulationResults import SimulationResults1D
from calibrationpinn.typeAliases import PLTFigure, PLTAxes, NPArray


PlotterConfig: TypeAlias = Domain1DPlotterConfig
SimulationResults: TypeAlias = SimulationResults1D


class Domain1DPlotter:
    def __init__(self, output_subdir_name: str, path_admin: PathAdministrator) -> None:
        self._output_subdir_name = output_subdir_name
        self._path_admin = path_admin
        self._fig_solution_and_prediction: Optional[PLTFigure] = None
        self._axes_solution_and_prediction: Optional[PLTAxes] = None
        self._fig_residual: Optional[PLTFigure] = None
        self._axes_residual: Optional[PLTAxes] = None
        self._fig_relative_error: Optional[PLTFigure] = None
        self._axes_relative_error: Optional[PLTAxes] = None

    def plot(
        self,
        simulation_results: SimulationResults,
        plotter_config: PlotterConfig,
    ) -> None:
        self._reset()
        self._set_up(simulation_results, plotter_config)
        self._plot(simulation_results, plotter_config)
        self._complete(plotter_config)
        self._save_plots(plotter_config)
        plt.close("all")

    def _reset(self) -> None:
        self._fig_solution_and_prediction = None
        self._axes_solution_and_prediction = None
        self._fig_residual = None
        self._axes_residual = None
        self._fig_relative_error = None
        self._axes_relative_error = None

    def _set_up(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ):
        self._set_up_figures_and_axes(simulation_results, plotter_config)
        # self._set_up_titles(plotter_config)

    def _set_up_figures_and_axes(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        (
            self._fig_solution_and_prediction,
            self._axes_solution_and_prediction,
        ) = self._set_up_one_figure_and_axes(simulation_results, plotter_config)
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
        x_rel_offset = 0.05
        coordinates = simulation_results.coordinates
        x_min = np.nanmin(coordinates)
        x_max = np.nanmax(coordinates)
        x_diff = x_max - x_min
        x_offset = x_rel_offset * x_diff
        x_min_with_offset = x_min - x_offset
        x_max_with_offset = x_max + x_offset
        axes.set_xlim(x_min_with_offset, x_max_with_offset)

    def _set_up_titles(self, plotter_config: PlotterConfig) -> None:
        self._set_up_title_for_one_plot(
            self._axes_solution_and_prediction,
            plotter_config.title_prefix_solution_and_prediction,
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
        title = self._combine_title(title_prefix, simulation_object)
        axes.set_title(title, pad=plotter_config.title_pad, **plotter_config.font)

    def _combine_title(self, title_prefix: str, simulation_object: str) -> str:
        return title_prefix + " " + simulation_object.lower()

    def _plot(
        self, simulation_results: SimulationResults, plotter_config: PlotterConfig
    ) -> None:
        self._plot_solution_and_prediction(
            simulation_results,
            plotter_config,
        )

        self._plot_errors(
            simulation_results.residuals,
            simulation_results,
            self._axes_residual,
            plotter_config.label_residual,
            plotter_config,
        )
        self._plot_errors(
            simulation_results.relative_errors,
            simulation_results,
            self._axes_relative_error,
            plotter_config.label_relative_error,
            plotter_config,
        )

    def _plot_solution_and_prediction(
        self,
        simulation_results: SimulationResults,
        plotter_config: PlotterConfig,
    ) -> None:
        self._axes_solution_and_prediction.plot(
            simulation_results.coordinates,
            simulation_results.solutions,
            plotter_config.format_string_solution,
            linewidth=3,
            markersize=9,
            label=plotter_config.label_solution,
            color="#bcbcbc",
        )
        self._axes_solution_and_prediction.plot(
            simulation_results.coordinates,
            simulation_results.predictions,
            plotter_config.format_string_prediction,
            linewidth=3,
            markersize=9,
            label=plotter_config.label_prediction,
            color="#2986cc",
        )

    def _plot_errors(
        self,
        errors: NPArray,
        simulation_results: SimulationResults,
        axes: PLTAxes,
        legend_label: str,
        plotter_config: PlotterConfig,
    ) -> None:
        axes.plot(
            simulation_results.coordinates,
            errors,
            plotter_config.format_string_errors,
            label=legend_label,
        )

    def _complete(self, plotter_config: PlotterConfig) -> None:
        self._set_up_one_legend(self._axes_solution_and_prediction, plotter_config)
        self._set_up_one_legend(self._axes_residual, plotter_config)
        self._set_up_one_legend(self._axes_relative_error, plotter_config)

    def _set_up_one_legend(self, axes: PLTAxes, plotter_config: PlotterConfig) -> None:
        axes.legend(fontsize=plotter_config.font_size, loc=0)

    def _save_plots(self, plotter_config: PlotterConfig) -> None:
        save_format = plotter_config.save_format
        self._save_one_plot(
            self._fig_solution_and_prediction,
            plotter_config.save_title_suffix_solution_and_prediction,
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
