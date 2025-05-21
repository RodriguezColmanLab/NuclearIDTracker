"""This plots the number of Paneth cells over time compared to the number of stem cells - during ablation and afterwards."""
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_ABLATION = "../../Data/Stem cell regeneration/Dataset - during DT treatment.autlist"
_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_REGENERATION_OFFSET_H = 16
_STEM_COLORS = ["#6D8660", "#A4CA90", "#C5E6B3"]  # From dark to light
_PANETH_COLORS = ["#916155", "#DA9180", "#F5B4A5"]
_ENTEROCYTE_COLORS = ["#304A5E", "#486F8D", "#7395B1"]


class _PredictedCellcount:
    stem_cell_count: List[int]
    paneth_cell_count: List[int]
    enterocyte_cell_count: List[int]
    goblet_cell_count: List[int]
    time_h: List[float]

    def __init__(self):
        self.stem_cell_count = list()
        self.paneth_cell_count = list()
        self.enterocyte_cell_count = list()
        self.goblet_cell_count = list()
        self.time_h = list()

    def add_entry(self, time_h: float, stem_cell_count: int, paneth_cell_count: int, enterocyte_cell_count: int,
                  goblet_cell_count: int):
        self.time_h.append(time_h)
        self.stem_cell_count.append(stem_cell_count)
        self.paneth_cell_count.append(paneth_cell_count)
        self.enterocyte_cell_count.append(enterocyte_cell_count)
        self.goblet_cell_count.append(goblet_cell_count)

    def offset_time(self, offset_h: float):
        self.time_h = [time + offset_h for time in self.time_h]

    def total_cell_count(self) -> List[int]:
        return [self.stem_cell_count[i] + self.paneth_cell_count[i] + self.enterocyte_cell_count[i] +
                self.goblet_cell_count[i] for i in range(len(self.time_h))]


def _count_predicted_paneth_cells(experiment: Experiment) -> _PredictedCellcount:
    timings = experiment.images.timings()
    predicted_cell_counts = _PredictedCellcount()
    for time_point in experiment.positions.time_points():
        time_h = timings.get_time_h_since_start(time_point)
        paneth_cell_count = 0
        stem_cell_count = 0
        enterocyte_cell_count = 0
        goblet_cell_count = 0
        for position in experiment.positions.of_time_point(time_point):
            cell_type = position_markers.get_position_type(experiment.position_data, position)
            if cell_type == "PANETH":
                paneth_cell_count += 1
            elif cell_type == "STEM":
                stem_cell_count += 1
            elif cell_type == "ENTEROCYTE":
                enterocyte_cell_count += 1
            elif cell_type == "MATURE_GOBLET":
                goblet_cell_count += 1
        predicted_cell_counts.add_entry(time_h, stem_cell_count, paneth_cell_count, enterocyte_cell_count,
                                         goblet_cell_count)
    return predicted_cell_counts


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

    figure = lib_figures.new_figure(size=(2.5, 3.8))
    _plot(figure, paneth_cells_ablation, paneth_cells_regeneration)
    figure.tight_layout()
    plt.show()


def _percentage(cell_counts: List[int], total_cell_counts: List[int]):
    """Returns the percentage of each cell type in the total cell count."""
    return [cell_counts[i] / total_cell_counts[i] * 100 if total_cell_counts[i] > 0 else 0 for i in range(len(cell_counts))]


def _plot(figure: Figure, paneth_cells_ablation: Dict[str, _PredictedCellcount], paneth_cells_regeneration: Dict[str, _PredictedCellcount]):
    ax_stem, ax_enterocyte, ax_paneth = figure.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
    useable_experiments = list(set(paneth_cells_ablation.keys()).intersection(set(paneth_cells_regeneration.keys())))
    print("Useable experiments: ", useable_experiments)

    for ax in [ax_stem, ax_enterocyte, ax_paneth]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvline(0, color="#b2bec3", linestyle="--", linewidth=3)
        ax.set_yticks(list(range(0, 101, 10)))

    ax_paneth.set_xlabel("Time (h)")

    regeneration_separator_x = _REGENERATION_OFFSET_H - 2

    ax_paneth.set_title("Paneth cells")
    ax_stem.set_title("Stem cells")
    ax_enterocyte.set_title("Enterocytes")

    ax_enterocyte.set_ylabel("Cells in organoid (%)")

    y_max = 80
    ax_stem.set_ylim(0, y_max)
    ax_enterocyte.set_ylim(0, y_max)
    y_max_paneth = 15
    ax_paneth.set_ylim(0, y_max_paneth)
    ax_stem.text(- 1.5, y_max, "DT treatment", ha="right", va="top")
    ax_stem.text(+ 1.5, y_max, "DT removal", ha="left", va="top")

    for i, name in enumerate(useable_experiments):
        ablation_experiment = paneth_cells_ablation[name]
        regeneration_experiment = paneth_cells_regeneration[name]

        times_h = ablation_experiment.time_h + regeneration_experiment.time_h
        times_h = [time_h - regeneration_separator_x for time_h in times_h]  # Move regeneration line to t=0
        paneth_cell_counts = ablation_experiment.paneth_cell_count + regeneration_experiment.paneth_cell_count
        stem_cell_counts = ablation_experiment.stem_cell_count + regeneration_experiment.stem_cell_count
        enterocyte_cell_counts = ablation_experiment.enterocyte_cell_count + regeneration_experiment.enterocyte_cell_count
        total_cell_counts = ablation_experiment.total_cell_count() + regeneration_experiment.total_cell_count()

        ax_paneth.plot(times_h, _percentage(paneth_cell_counts, total_cell_counts), color=_PANETH_COLORS[i], linewidth=2)
        ax_stem.plot(times_h, _percentage(stem_cell_counts, total_cell_counts), color=_STEM_COLORS[i], linewidth=2)
        ax_enterocyte.plot(times_h, _percentage(enterocyte_cell_counts, total_cell_counts), color=_ENTEROCYTE_COLORS[i], linewidth=2)


if __name__ == "__main__":
    main()