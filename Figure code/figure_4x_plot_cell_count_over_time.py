from typing import List

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
import lib_figures
from organoid_tracker.core import Name
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_MAX_TIME_H = 230


class _CellCountsOverTime:
    """Shows the number of cells of different types over time for a single experiment."""
    experiment_name: Name
    times_h: List[float]
    stem_cell_counts: List[int]
    enterocyte_counts: List[int]
    paneth_cell_counts: List[int]

    def __init__(self, experiment_name: Name):
        self.experiment_name = experiment_name
        self.times_h = []
        self.stem_cell_counts = []
        self.enterocyte_counts = []
        self.paneth_cell_counts = []

    def add_time_point(self, time_h: float, stem_cell_count: int, enterocyte_count: int, paneth_cell_count: int):
        if time_h > _MAX_TIME_H:
            return
        self.times_h.append(time_h)
        self.stem_cell_counts.append(stem_cell_count)
        self.enterocyte_counts.append(enterocyte_count)
        self.paneth_cell_counts.append(paneth_cell_count)


def _get_cell_counts_over_time(experiment: Experiment) -> _CellCountsOverTime:
    timings = experiment.images.timings()
    cell_counts = _CellCountsOverTime(experiment.name)
    for time_point in experiment.positions.time_points():
        stem_cell_count = 0
        enterocyte_count = 0
        paneth_cell_count = 0
        for position in experiment.positions.of_time_point(time_point):
            cell_type = position_markers.get_position_type(experiment.position_data, position)
            if cell_type == "STEM":
                stem_cell_count += 1
            elif cell_type == "ENTEROCYTE":
                enterocyte_count += 1
            elif cell_type == "PANETH":
                paneth_cell_count += 1
        cell_counts.add_time_point(
            time_h=timings.get_time_h_since_start(time_point),
            stem_cell_count=stem_cell_count,
            enterocyte_count=enterocyte_count,
            paneth_cell_count=paneth_cell_count
        )
    return cell_counts


def main():
    cell_count_regeneration = list()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_REGENERATION, load_images=False):
        cell_count_regeneration.append(_get_cell_counts_over_time(experiment))

    cell_count_control = list()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_CONTROL, load_images=False):
        cell_count_control.append(_get_cell_counts_over_time(experiment))

    columns = max(len(cell_count_control), len(cell_count_regeneration))

    figure = lib_figures.new_figure(size=(10, 10))
    axes = figure.subplots(nrows=2, ncols=columns, sharex=True, sharey=True)
    for i in range(len(cell_count_regeneration)):
        _draw_plot(axes[0, i], cell_count_regeneration[i])
    axes[0, columns // 2].set_title("Regeneration")
    for i in range(len(cell_count_control)):
        _draw_plot(axes[1, i], cell_count_control[i])
    axes[1, columns // 2].set_title("Control")
    plt.show()


def _draw_plot(ax: Axes, cell_count_over_time: _CellCountsOverTime):
    ax.fill_between(cell_count_over_time.times_h, [0] * len(cell_count_over_time.times_h),
                    cell_count_over_time.enterocyte_counts, label="Enterocytes", color=lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"])
    ax.fill_between(cell_count_over_time.times_h, cell_count_over_time.enterocyte_counts,
                    numpy.array(cell_count_over_time.enterocyte_counts) + cell_count_over_time.stem_cell_counts,
                    label="Stem cells", color=lib_figures.CELL_TYPE_PALETTE["STEM"])
    ax.legend()

    # Add organoid name to top left of axis
    ax.text(0.05, 0.95, cell_count_over_time.experiment_name.get_name(),
            transform=ax.transAxes, fontsize=7, verticalalignment='top')


if __name__ == "__main__":
    main()