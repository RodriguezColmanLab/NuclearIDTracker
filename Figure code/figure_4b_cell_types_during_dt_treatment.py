from collections import defaultdict
from typing import List, Dict

import matplotlib.colors
import numpy
from numpy import ndarray
from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core import Name, MPLColor
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATASET_FILE = "../../Data/Stem cell regeneration/Dataset - during DT treatment.autlist"


class _CellCountsOverTime:
    """Shows the number of cells of different types over time for a single experiment."""
    experiment_name: Name
    times_h: List[float]
    counts_per_cell_type: Dict[str, List[int]]

    def __init__(self, experiment_name: Name):
        self.experiment_name = experiment_name
        self.times_h = []
        self.counts_per_cell_type = defaultdict(list)

    def add_time_point(self, time_h: float, count_per_cell_type: Dict[str, int]):
        self.times_h.append(time_h)
        for cell_type, count in count_per_cell_type.items():
            self.counts_per_cell_type[cell_type].append(count)

        # Check if all cell type arrays have the same length
        time_point_count = len(self.times_h)
        for cell_type, counts in self.counts_per_cell_type.items():
            if len(counts) < time_point_count:
                raise ValueError(f"Cell type {cell_type} has less counts than the other cell types.")

    def total_cell_counts(self) -> ndarray:
        result = numpy.zeros(len(self.times_h), dtype=int)
        for counts in self.counts_per_cell_type.values():
            result += counts
        return result



def _get_cell_counts_over_time(experiment: Experiment) -> _CellCountsOverTime:
    timings = experiment.images.timings()
    cell_counts = _CellCountsOverTime(experiment.name)
    cell_types = experiment.global_data.get_data("ct_probabilities")
    for time_point in experiment.positions.time_points():
        count_per_cell_type = {cell_type: 0 for cell_type in cell_types}
        for position in experiment.positions.of_time_point(time_point):
            cell_type = position_markers.get_position_type(experiment.position_data, position)
            if cell_type in cell_types:
                count_per_cell_type[cell_type] += 1
        cell_counts.add_time_point(
            time_h=timings.get_time_h_since_start(time_point),
            count_per_cell_type=count_per_cell_type
        )
    return cell_counts


def _adjust_color(color: MPLColor, experiment_index: int) -> MPLColor:
    # Convert to RGB color between 0 and 1
    r, g, b = matplotlib.colors.to_rgb(color)

    if experiment_index == 0:
        return r * 0.3, g * 0.3, b * 0.3

    return r, g, b


def main():
    cell_counts = []
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE, load_images=False):
        cell_counts.append(_get_cell_counts_over_time(experiment))

    # Plot the data
    figure = lib_figures.new_figure(size=(5.5, 2))
    ax_stem, ax_enterocyte, ax_paneth = figure.subplots(nrows=1, ncols=3, sharex=True, sharey=False)
    ax_stem.set_ylabel("Stem cells (%)")
    ax_enterocyte.set_ylabel("Enterocytes (%)")
    ax_paneth.set_ylabel("Paneth cells (%)")
    for i, cell_count in enumerate(cell_counts):
        print("Organoid", i + 1, ":", cell_count.experiment_name)
        total_cell_counts = cell_count.total_cell_counts()
        ax_stem.plot(cell_count.times_h, cell_count.counts_per_cell_type["STEM"] / total_cell_counts * 100,
                     color=_adjust_color(lib_figures.CELL_TYPE_PALETTE["STEM"], i),
                     label=f"Organoid {i + 1}", linewidth=2, alpha=0.5 if i == 2 else 1)
        ax_enterocyte.plot(cell_count.times_h, cell_count.counts_per_cell_type["ENTEROCYTE"] / total_cell_counts * 100,
                           color=_adjust_color(lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"], i),
                           label=f"Organoid {i + 1}", linewidth=2, alpha=0.5 if i == 2 else 1)
        ax_paneth.plot(cell_count.times_h, cell_count.counts_per_cell_type["PANETH"] / total_cell_counts * 100,
                       color=_adjust_color(lib_figures.CELL_TYPE_PALETTE["PANETH"], i),
                       label=f"Organoid {i + 1}", linewidth=2, alpha=0.5 if i == 2 else 1)
    ax_stem.set_xlabel("Time (h)")
    ax_enterocyte.set_xlabel("Time (h)")
    ax_paneth.set_xlabel("Time (h)")
    ax_stem.legend()
    ax_enterocyte.legend()
    ax_paneth.legend()

    # Set y-axis limits and ticks
    y_lim_high = 100  # Used for the stem cells and enterocytes
    y_lim_low = 25  # Used for the Paneth cells
    ax_stem.set_ylim(0, y_lim_high)
    ax_stem.set_yticks(numpy.arange(0, y_lim_high + 1, 5), minor=True)
    ax_stem.set_yticks(numpy.arange(0, y_lim_high + 1, 25), minor=False)
    ax_enterocyte.set_ylim(0, y_lim_high)
    ax_enterocyte.set_yticks(numpy.arange(0, y_lim_high + 1, 5), minor=True)
    ax_enterocyte.set_yticks(numpy.arange(0, y_lim_high + 1, 25), minor=False)
    ax_paneth.set_ylim(0, y_lim_low)
    ax_paneth.set_yticks(numpy.arange(0, y_lim_low + 1, 5), minor=True)
    ax_paneth.set_yticks(numpy.arange(0, y_lim_low + 1, 25), minor=False)

    # Hide top and right axis
    ax_stem.spines['top'].set_visible(False)
    ax_stem.spines['right'].set_visible(False)
    ax_enterocyte.spines['top'].set_visible(False)
    ax_enterocyte.spines['right'].set_visible(False)
    ax_paneth.spines['top'].set_visible(False)
    ax_paneth.spines['right'].set_visible(False)

    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
