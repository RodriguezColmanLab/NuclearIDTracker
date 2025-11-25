"""This plots the number of Paneth cells over time compared to the number of stem cells - during ablation and afterwards."""
from typing import List, Dict

from matplotlib.figure import Figure

import lib_figures

import matplotlib.pyplot as plt

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.util.moving_average import MovingAverage

_DATA_FILE_ABLATION = "../../Data/Stem cell regeneration/Dataset - during DT treatment.autlist"
_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_REGENERATION_OFFSET_H = 16
_COLORS = ["#916155", "#DA9180", "#F5B4A5"]  # From dark to light


class _CellCount:
    cell_count: List[int]
    time_h: List[float]

    def __init__(self):
        self.cell_count = list()
        self.time_h = list()

    def add_entry(self, time_h: float, cell_count: int):
        self.time_h.append(time_h)
        self.cell_count.append(cell_count)

    def offset_time(self, offset_h: float):
        self.time_h = [time + offset_h for time in self.time_h]


def _count_cells(experiment: Experiment) -> _CellCount:
    timings = experiment.images.timings()
    cell_count = _CellCount()
    for time_point in experiment.positions.time_points():
        time_h = timings.get_time_h_since_start(time_point)
        cell_count.add_entry(time_h, len(experiment.positions.of_time_point(time_point)))
    return cell_count


def main():
    cell_counts_ablation = dict()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_ABLATION):
        name = experiment.name.get_name().replace("add", "").strip()
        cell_counts_ablation[name] = _count_cells(experiment)

    cell_counts_regeneration = dict()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_REGENERATION):
        name = experiment.name.get_name().replace("remove", "").strip()
        counts = _count_cells(experiment)
        counts.offset_time(_REGENERATION_OFFSET_H)
        cell_counts_regeneration[name] = counts

    with open("cell_counts_ablation_regeneration.txt", "w") as handle:
        useable_experiments = set(cell_counts_ablation.keys()).intersection(set(cell_counts_regeneration.keys()))
        for i, name in enumerate(useable_experiments):
            ablation_experiment = cell_counts_ablation[name]
            regeneration_experiment = cell_counts_regeneration[name]
            regeneration_separator_x = _REGENERATION_OFFSET_H - 2

            times_h = ablation_experiment.time_h + regeneration_experiment.time_h
            times_h = [time_h - regeneration_separator_x for time_h in times_h]  # Move regeneration line to t=0
            cell_counts = ablation_experiment.cell_count + regeneration_experiment.cell_count

            handle.write(f"{name}\n")
            handle.write(f"Times (h),"+ ','.join(map(str, times_h)) + "\n")
            handle.write(f"Cell counts," + ','.join(map(str, cell_counts)) + "\n")
            handle.write("\n")



    figure = lib_figures.new_figure(size=(2.8, 2))
    _plot(figure, cell_counts_ablation, cell_counts_regeneration)
    figure.tight_layout()
    plt.show()


def _plot(figure: Figure, paneth_cells_ablation: Dict[str, _CellCount], paneth_cells_regeneration: Dict[str, _CellCount]):
    ax = figure.gca()
    useable_experiments = set(paneth_cells_ablation.keys()).intersection(set(paneth_cells_regeneration.keys()))

    y_max = 1500
    ax.set_ylim(0, y_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Cell count")
    ax.set_xlabel("Time (h)")

    regeneration_separator_x = _REGENERATION_OFFSET_H - 2
    ax.axvline(0, color="#b2bec3", linestyle="--", linewidth=3)
    ax.text(- 1.5, y_max, "DT treatment", ha="right", va="top")
    ax.text(+ 1.5, y_max, "DT removal", ha="left", va="top")

    for i, name in enumerate(useable_experiments):
        ablation_experiment = paneth_cells_ablation[name]
        regeneration_experiment = paneth_cells_regeneration[name]

        times_h = ablation_experiment.time_h + regeneration_experiment.time_h
        times_h = [time_h - regeneration_separator_x for time_h in times_h]  # Move regeneration line to t=0
        cell_counts = ablation_experiment.cell_count + regeneration_experiment.cell_count

        ax.scatter(times_h, cell_counts,  s=6 + 2 * i, linewidths=0, color=_COLORS[i], alpha=0.7)


if __name__ == "__main__":
    main()