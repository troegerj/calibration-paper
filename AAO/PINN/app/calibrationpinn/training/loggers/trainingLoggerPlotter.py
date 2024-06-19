# Standard library imports
from pathlib import Path
from typing import Optional, TYPE_CHECKING

# Third library imports
import matplotlib.pyplot as plt

# Local library imports
from calibrationpinn.inputoutput import PathAdministrator
from calibrationpinn.training.loggers.trainingLogger import TrainingLogger
from calibrationpinn.training.loggers.trainingLoggerPlotterConfig import (
    TrainingLoggerPlotterConfig,
)
from calibrationpinn.typeAliases import PLTFigure, PLTAxes


class TrainingLoggerPlotter:
    def __init__(
        self, output_subdir_name: Optional[str], path_admin: PathAdministrator
    ) -> None:
        self._output_subdir_name = output_subdir_name
        self._path_admin = path_admin
        self._fig_losses: Optional[PLTFigure] = None
        self._axes_losses: Optional[PLTAxes] = None
        self._fig_errors: Optional[PLTFigure] = None
        self._axes_errors: Optional[PLTAxes] = None
        self._fig_params: Optional[PLTFigure] = None
        self._axes_params: Optional[PLTAxes] = None

    def plot(
        self,
        training_logger: TrainingLogger,
        plotter_config_losses: TrainingLoggerPlotterConfig,
        plotter_config_errors: TrainingLoggerPlotterConfig,
        plotter_config_params: TrainingLoggerPlotterConfig,
    ) -> None:
        self._set_up(
            plotter_config_losses, plotter_config_errors, plotter_config_params
        )
        self._plot_training_logger(training_logger)
        self._complete(
            plotter_config_losses, plotter_config_errors, plotter_config_params
        )
        self._save_plots(
            plotter_config_losses, plotter_config_errors, plotter_config_params
        )
        plt.show()

    def _set_up(
        self,
        plotter_config_losses: TrainingLoggerPlotterConfig,
        plotter_config_errors: TrainingLoggerPlotterConfig,
        plotter_config_params: TrainingLoggerPlotterConfig,
    ) -> None:
        self._reset()
        self._set_up_plots(
            plotter_config_losses, plotter_config_errors, plotter_config_params
        )

    def _reset(self) -> None:
        self._fig_losses = None
        self._axes_losses = None
        self._fig_errors = None
        self._axes_errors = None
        self._fig_params = None
        self._axes_params = None

    def _set_up_plots(
        self,
        plotter_config_losses: TrainingLoggerPlotterConfig,
        plotter_config_errors: TrainingLoggerPlotterConfig,
        plotter_config_params: TrainingLoggerPlotterConfig,
    ) -> None:
        (
            self._fig_losses,
            self._axes_losses,
        ) = self._set_up_one_plot_with_logaritmic_axis(plotter_config_losses)
        (
            self._fig_errors,
            self._axes_errors,
        ) = self._set_up_one_plot_with_logaritmic_axis(plotter_config_errors)
        self._fig_params, self._axes_params = self._set_up_one_plot(
            plotter_config_params
        )

    def _set_up_one_plot_with_logaritmic_axis(
        self, plotter_config: TrainingLoggerPlotterConfig
    ) -> tuple[PLTFigure, PLTAxes]:
        figure, axes = self._set_up_figure_and_axes(plotter_config)
        self._set_up_title(axes, plotter_config)
        self._set_up_logarithmic_axis(axes, plotter_config)
        return figure, axes

    def _set_up_one_plot(
        self, plotter_config: TrainingLoggerPlotterConfig
    ) -> tuple[PLTFigure, PLTAxes]:
        figure, axes = self._set_up_figure_and_axes(plotter_config)
        # self._set_up_title(axes, plotter_config)
        self._set_up_axis(axes, plotter_config)
        return figure, axes

    def _set_up_figure_and_axes(
        self, plotter_config: TrainingLoggerPlotterConfig
    ) -> tuple[PLTFigure, PLTAxes]:
        figure, axes = plt.subplots()
        self._set_up_axis_labels(axes, plotter_config)
        return figure, axes

    def _set_up_title(self, axes, plotter_config: TrainingLoggerPlotterConfig) -> None:
        axes.set_title(plotter_config.title, **plotter_config.font)

    def _set_up_logarithmic_axis(
        self, axes: PLTAxes, plotter_config: TrainingLoggerPlotterConfig
    ) -> None:
        self._set_up_axis_labels(axes, plotter_config)
        axes.set_yscale("log")
        axes.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        axes.xaxis.offsetText.set_fontsize(plotter_config.scientific_notation_size)
        axes.yaxis.offsetText.set_fontsize(plotter_config.scientific_notation_size)

    def _set_up_axis(self, axes: PLTAxes, plotter_config: TrainingLoggerPlotterConfig):
        self._set_up_axis_labels(axes, plotter_config)
        axes.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        axes.xaxis.offsetText.set_fontsize(plotter_config.scientific_notation_size)
        axes.yaxis.offsetText.set_fontsize(plotter_config.scientific_notation_size)

    def _set_up_axis_labels(
        self, axes: PLTAxes, plotter_config: TrainingLoggerPlotterConfig
    ) -> None:
        axes.set_xlabel(plotter_config.x_label, **plotter_config.font)
        axes.set_ylabel(plotter_config.y_label, **plotter_config.font)
        axes.tick_params(
            axis="both", which="minor", labelsize=plotter_config.minor_tick_label_size
        )
        axes.tick_params(
            axis="both", which="major", labelsize=plotter_config.major_tick_label_size
        )

    def _plot_training_logger(self, training_logger: TrainingLogger) -> None:
        self._plot_losses(training_logger)
        self._plot_errors(training_logger)
        self._plot_params(training_logger)

    def _plot_losses(self, training_logger: TrainingLogger) -> None:
        epochs = training_logger.epochs_record.as_data_frame()
        losses = training_logger.losses_record.as_data_frame()
        loss_names = list(losses.columns.values)
        for loss_name in loss_names:
            color, loss_name_label = self._select_color_and_label_for_loss_plot(
                loss_name
            )
            self._axes_losses.plot(
                epochs.to_numpy(),
                losses[loss_name].to_numpy(),
                color=color,
                linewidth=3,
                label=loss_name_label,
            )

    def _plot_errors(self, training_logger: TrainingLogger) -> None:
        epochs = training_logger.epochs_record.as_data_frame()
        errors = training_logger.error_metrics_record.as_data_frame()
        error_names = list(errors.columns.values)
        for error_name in error_names:
            self._axes_errors.plot(
                epochs.to_numpy(), errors[error_name].to_numpy(), label=error_name
            )

    def _plot_params(self, training_logger: TrainingLogger) -> None:
        epochs = training_logger.epochs_record.as_data_frame()
        params = training_logger.parameters_record.as_data_frame()
        param_names = list(params.columns.values)
        for param_name in param_names:
            color, param_name_label = self._select_color_and_label_for_params_plot(
                param_name
            )
            self._axes_params.plot(
                epochs.to_numpy(),
                params[param_name].to_numpy(),
                color=color,
                linewidth=3,
                label=param_name_label,
            )

    def _select_color_and_label_for_loss_plot(self, loss_name) -> tuple[str, str]:
        loss_name_label = loss_name
        color = "black"
        if loss_name == "loss_pde":
            loss_name_label = r"$\mathcal{\lambda}_{pde}$"
            color = "#440154"
        elif loss_name == "loss_uy":
            loss_name_label = r"$\mathcal{\lambda}_{o_{y}}$"
            color = "#3b528b"
        elif loss_name == "loss_ux" or loss_name == "loss_data":
            loss_name_label = r"$\mathcal{\lambda}_{o_{x}}$"
            color = "#21918c"
        elif loss_name == "loss_energy":
            loss_name_label = r"$\mathcal{\lambda}_{W}$"
            color = "#5ec962"
        return color, loss_name_label

    def _select_color_and_label_for_params_plot(self, param_name) -> tuple[str, str]:
        param_name_label = param_name
        color = "black"
        if param_name == "K_cor":
            param_name_label = r"$\alpha_{K}$"
            color = "#440154"
        elif param_name == "E_cor":
            param_name_label = r"$\alpha_{E}$"
            color = "#440154"
        elif param_name == "G_cor":
            param_name_label = r"$\alpha_{G}$"
            color = "#21918c"
        elif param_name == "nu_cor":
            param_name_label = r"$\alpha_{\nu}$"
            color = "#21918c"
        return color, param_name_label

    def _complete(
        self,
        plotter_config_losses: TrainingLoggerPlotterConfig,
        plotter_config_errors: TrainingLoggerPlotterConfig,
        plotter_config_params: TrainingLoggerPlotterConfig,
    ) -> None:
        self._set_up_legend(self._axes_losses, plotter_config_losses)
        self._set_up_legend(self._axes_errors, plotter_config_errors)
        self._set_up_legend(self._axes_params, plotter_config_params)

    def _set_up_legend(
        self, axes: PLTAxes, plotter_config: TrainingLoggerPlotterConfig
    ) -> None:
        axes.legend(fontsize=plotter_config.font_size)

    def _save_plots(
        self,
        plotter_config_losses: TrainingLoggerPlotterConfig,
        plotter_config_errors: TrainingLoggerPlotterConfig,
        plotter_config_params: TrainingLoggerPlotterConfig,
    ) -> None:
        self._save_one_plot(self._fig_losses, plotter_config_losses)
        self._save_one_plot(self._fig_errors, plotter_config_errors)
        self._save_one_plot(self._fig_params, plotter_config_params)

    def _save_one_plot(
        self, fig: PLTFigure, plotter_config: TrainingLoggerPlotterConfig
    ) -> None:
        save_format = plotter_config.save_format
        plot_name = plotter_config.save_title
        file_name = self._set_file_name_for_plot(plot_name, save_format)
        save_path = self._get_save_path(file_name)
        fig.savefig(save_path, format=save_format, bbox_inches="tight")

    def _set_file_name_for_plot(self, plot_name: str, save_format: str) -> str:
        return "plot_{plot_name}.{file_ending}".format(
            plot_name=plot_name, file_ending=save_format
        )

    def _get_save_path(self, file_name: str) -> Path:
        return self._path_admin.get_path_to_output_file(
            file_name, self._output_subdir_name
        )
