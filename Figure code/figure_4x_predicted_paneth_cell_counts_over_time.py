"""This plots the number of Paneth cells over time compared to the number of stem cells - during ablation and afterwards."""
import numpy
from typing import List, Dict

from matplotlib.figure import Figure

import lib_figures

import matplotlib.pyplot as plt

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.util.moving_average import MovingAverage
import lib_data

_DATA_FILE_ABLATION = "../../Data/Stem cell regeneration/Dataset - during DT treatment.autlist"
_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_REGENERATION_OFFSET_H = 16
_COLORS = ["#916155", "#DA9180", "#F5B4A5"]  # From dark to light


class _PanethCellCount:
    stem_cell_count: List[int]
    paneth_cell_count: List[int]
    time_h: List[float]

    def __init__(self):
        self.stem_cell_count = list()
        self.paneth_cell_count = list()
        self.time_h = list()

    def add_entry(self, time_h: float, stem_cell_count: int, paneth_cell_count: int):
        self.time_h.append(time_h)
        self.stem_cell_count.append(stem_cell_count)
        self.paneth_cell_count.append(paneth_cell_count)

    def offset_time(self, offset_h: float):
        self.time_h = [time + offset_h for time in self.time_h]


def _count_predicted_paneth_cells(experiment: Experiment) -> _PanethCellCount:
    timings = experiment.images.timings()
    cell_types = experiment.global_data.get_data("ct_probabilities")
    paneth_cell_counts = _PanethCellCount()
    for time_point in experiment.positions.time_points():
        time_h = timings.get_time_h_since_start(time_point)
        paneth_cell_count = 0
        stem_cell_count = 0
        for position in experiment.positions.of_time_point(time_point):
            ct_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
            stem_to_ec_location = lib_data.find_stem_to_ec_location(cell_types, ct_probabilities)
            cell_type = position_markers.get_position_type(experiment.position_data, position)
            if cell_type == "PANETH":
                paneth_cell_count += 1
            elif stem_to_ec_location is not None and stem_to_ec_location > 0.55:
                stem_cell_count += 1
        paneth_cell_counts.add_entry(time_h, stem_cell_count, paneth_cell_count)
    return paneth_cell_counts


def main():
    paneth_cells_ablation = dict()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_ABLATION):
        name = experiment.name.get_name().replace("add", "").strip()
        paneth_cells_ablation[name] = _count_predicted_paneth_cells(experiment)

    paneth_cells_regeneration = dict()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_REGENERATION):
        name = experiment.name.get_name().replace("remove", "").strip()
        counts = _count_predicted_paneth_cells(experiment)
        counts.offset_time(_REGENERATION_OFFSET_H)
        paneth_cells_regeneration[name] = counts

    figure = lib_figures.new_figure(size=(2.8, 2))
    _plot(figure, paneth_cells_ablation, paneth_cells_regeneration)
    figure.tight_layout()
    plt.show()


def _plot(figure: Figure, paneth_cells_ablation: Dict[str, _PanethCellCount], paneth_cells_regeneration: Dict[str, _PanethCellCount]):
    ax = figure.gca()
    useable_experiments = set(paneth_cells_ablation.keys()).intersection(set(paneth_cells_regeneration.keys()))

    y_max = 3.33
    ax.set_ylim(0, y_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Predicted Paneth / stem cell count")
    ax.set_xlabel("Time (h)")

    regeneration_separator_x = _REGENERATION_OFFSET_H - 2
    ax.axvline(0, color="#b2bec3", linestyle="--", linewidth=3)
    ax.text(- 1.5, y_max, "DT treatment", ha="right", va="top")
    ax.text(+ 1.5, y_max, "DT removal", ha="left", va="top")


    all_times_h = []
    all_paneth_over_stem = []
    for i, name in enumerate(useable_experiments):
        ablation_experiment = paneth_cells_ablation[name]
        regeneration_experiment = paneth_cells_regeneration[name]

        times_h = ablation_experiment.time_h + regeneration_experiment.time_h
        times_h = [time_h - regeneration_separator_x for time_h in times_h]  # Move regeneration line to t=0
        paneth_cell_counts = ablation_experiment.paneth_cell_count + regeneration_experiment.paneth_cell_count
        stem_cell_counts = ablation_experiment.stem_cell_count + regeneration_experiment.stem_cell_count
        paneth_over_stem = [paneth / stem for paneth, stem in zip(paneth_cell_counts, stem_cell_counts)]


        ax.scatter(times_h, paneth_over_stem,  s=6 + 2 * i, linewidths=0, color=_COLORS[i], alpha=0.7)

        all_times_h += times_h
        all_paneth_over_stem += paneth_over_stem

    all_times_h = numpy.array(all_times_h)
    all_paneth_over_stem = numpy.array(all_paneth_over_stem)

    # Sort the data for moving average
    sorted_indices = numpy.argsort(all_times_h)
    all_times_h = all_times_h[sorted_indices]
    all_paneth_over_stem = all_paneth_over_stem[sorted_indices]

    moving_average = MovingAverage(all_times_h, all_paneth_over_stem, window_width=4.5, x_step_size=0.2)
    ax.plot(moving_average.x_values, moving_average.mean_values, color=_COLORS[0], linewidth=3, label="Average")


if __name__ == "__main__":
    main()