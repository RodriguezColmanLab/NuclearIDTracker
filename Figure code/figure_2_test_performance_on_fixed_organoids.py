from typing import List

import matplotlib.colors
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_OLFM4_STAINING_AND_PREDICTIONS = "../../Data/Fixed cells control/Olfm4 staining.autlist"

_DATA_FILE_KRT20_LYZ_PREDICTIONS = "../../Data/Testing data - predictions - treatments - fixed.autlist"
_DATA_FILE_KRT20_LYZ_STAINING = "../../Data/Immunostaining conditions.autlist"

class _PredictedCellTypes:

    stained_type: str
    predicted_cell_types: List[str]

    def __init__(self, stained_type: str):
        self.stained_type = stained_type
        self.predicted_cell_types = list()

    def add_entry(self, predicted_cell_type: str):
        self.predicted_cell_types.append(predicted_cell_type)

    def get_fraction(self, cell_type: str) -> float:
        if len(self.predicted_cell_types) == 0:
            return 0.0
        return self.predicted_cell_types.count(cell_type) / len(self.predicted_cell_types)


def _plot_predictions(ax: Axes, predicted_cell_types: List[_PredictedCellTypes]):
    # Collect all used cell types
    all_predicted_cell_types = set()
    for cell_type in predicted_cell_types:
        all_predicted_cell_types.update(cell_type.predicted_cell_types)
    all_predicted_cell_types = sorted(all_predicted_cell_types)

    # Make a bar plot
    bottom = numpy.zeros(len(predicted_cell_types), dtype=float)
    x_values = list(range(len(predicted_cell_types)))
    for cell_type in all_predicted_cell_types:
        percentages = [predicted_cell_type.get_fraction(cell_type) * 100 for predicted_cell_type in predicted_cell_types]
        color = lib_figures.CELL_TYPE_PALETTE[cell_type]
        ax.bar(x_values, percentages, bottom=bottom, label=lib_figures.style_cell_type_name(cell_type), color=color)

        # Add percentages
        for i, predicted_cell_type in enumerate(predicted_cell_types):
            if percentages[i] < 0.5:
                continue
            # Make sure the text is readable
            color = "black"
            if cell_type == "ENTEROCYTE":
                color = "white"
            ax.text(i, bottom[i] + percentages[i] / 2, f"{percentages[i]:.1f}%", ha="center", va="center", color=color)

        bottom += percentages


    # Add counts
    for i, predicted_cell_type in enumerate(predicted_cell_types):
        ax.text(i, 103, str(len(predicted_cell_type.predicted_cell_types)), ha="center", va="bottom")


def main():

    # Collect data for OLFM4 staining (here predictions and staining are in the same file)
    olfm4_cells = _PredictedCellTypes("STEM")
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_OLFM4_STAINING_AND_PREDICTIONS, load_images=False):
        cell_types = experiment.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise ValueError(f"No cell type predictions found in the experiment {experiment.name}.")

        for position in experiment.positions:
            stained_type = position_markers.get_position_type(experiment.position_data, position)
            if stained_type != "STEM":
                continue
            predicted_scores = experiment.position_data.get_position_data(position, "ct_probabilities")
            if predicted_scores is None:
                continue
            predicted_cell_type = cell_types[numpy.argmax(predicted_scores)]
            olfm4_cells.add_entry(predicted_cell_type)

    # Collect data for other stainings (here predictions and staining are in different files)
    krt20_cells = _PredictedCellTypes("ENTEROCYTE")
    wga_cells = _PredictedCellTypes("PANETH")
    experiments_staining = list(list_io.load_experiment_list_file(_DATA_FILE_KRT20_LYZ_STAINING, load_images=False))
    experiments_predictions = list(list_io.load_experiment_list_file(_DATA_FILE_KRT20_LYZ_PREDICTIONS, load_images=False))
    for experiment_staining, experiment_predictions in zip(experiments_staining, experiments_predictions):
        if experiment_staining.name.get_name() != experiment_predictions.name.get_name():
            raise ValueError("The autlist files do not match: " + experiment_staining.name.get_name() +
                             " and " + experiment_predictions.name.get_name())
        if "control" not in experiment_staining.name.get_name():
            continue  # We don't use the treated organoids for this figure

        cell_types = experiment_predictions.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise ValueError(f"No cell type predictions found in the experiment {experiment_predictions.name}.")

        for position in experiment_staining.positions:
            stained_type = position_markers.get_position_type(experiment_staining.position_data, position)
            if stained_type is None:
                continue
            predicted_scores = experiment_predictions.position_data.get_position_data(position, "ct_probabilities")
            if predicted_scores is None:
                continue
            predicted_cell_type = cell_types[numpy.argmax(predicted_scores)]
            if stained_type == "ENTEROCYTE":
                krt20_cells.add_entry(predicted_cell_type)
            elif stained_type == "PANETH":
                wga_cells.add_entry(predicted_cell_type)

    figure = lib_figures.new_figure()
    ax = figure.gca()
    _plot_predictions(ax, [olfm4_cells, wga_cells, krt20_cells])
    ax.set_xlim(-0.7, 2.6)
    ax.set_ylim(0, 110)
    ax.set_xticks([0, 1, 2], ["Olfm4+ cells", "Lyz+ cells", "KRT20+ cells"], rotation=-45, ha="left")
    ax.set_ylabel("Predicted cell type (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
