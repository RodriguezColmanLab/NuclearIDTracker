"""We plot all stem and Paneth cells on a stem-to-Paneth axis, and see how their amount changes over time."""
from typing import List, Callable, Optional

import numpy
from matplotlib import pyplot as plt

import lib_data
import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

_DATA_FILE = "../../Data/Tracking data as controls/Dataset - full overweekend.autlist"
_TIME_INTERVAL_H = 15


class _StemToXData:
    """Stores the stemness of cells along the stem-to-X (Paneth, enterocyte) axis, for multiple time points."""
    _bins: List[List[float]]

    def __init__(self):
        self._bins = list()

    def add_values(self, time_h: float, values: List[float]):
        if time_h < 0:
            raise ValueError("Time must be non-negative")
        bin_index = int(time_h / _TIME_INTERVAL_H)
        while len(self._bins) <= bin_index:
            self._bins.append(list())
        self._bins[bin_index].extend(values)

    def times_h(self) -> List[float]:
        return [_TIME_INTERVAL_H * i for i in range(len(self._bins))]

    def values(self) -> List[List[float]]:
        return self._bins


def main():
    stem_to_enterocyte_data = _StemToXData()

    for experiment in list_io.load_experiment_list_file(_DATA_FILE, load_images=False):
        _collect_experiment_data(experiment, into=stem_to_enterocyte_data, cell_to_axis_function=lib_data.find_stem_to_ec_location)

    figure = lib_figures.new_figure()
    times_h = stem_to_enterocyte_data.times_h()
    all_stem_to_ec_values = stem_to_enterocyte_data.values()

    axes = figure.subplots(nrows=len(times_h), ncols=1, sharex=False, sharey="col")
    for i, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


        # Plot stem bins
        _, bins, patches = ax.hist(all_stem_to_ec_values[i], bins=numpy.arange(0, 1, 0.02))
        for j in range(len(bins) - 1):
            bin_x = (bins[j] + bins[j + 1]) / 2
            patches[j].set_facecolor(lib_figures.get_stem_to_ec_color(bin_x))

        ax.set_xlim(0.8, 0.1)

        if i != len(times_h) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
        else:
            ax.set_xticks(numpy.arange(min(ax.get_xlim()), max(ax.get_xlim()) + 0.01, 0.1))

        if i == len(times_h) - 1:
            ax.set_xlabel("Stem-to-enterocyte axis")

        # Set y scale for stem and enterocyte (they are the same)
        ax.set_yscale("log")
        ax.set_ylim(8, 700)
        ax.set_yticks([10, 100])
        ax.set_yticklabels(["10", "100"])

        # Add time to top left of panel
        ax.text(1.0, 0.8, f"{times_h[i]}h", transform=ax.transAxes, horizontalalignment="right",
                verticalalignment="top")

    axes[len(axes) // 2].set_ylabel("Cell count")
    plt.show()


def _collect_experiment_data(experiment: Experiment, *, into: _StemToXData,
                             cell_to_axis_function: Callable[[List[str], Optional[List[float]]], Optional[float]]):
    cell_types = experiment.global_data.get_data("ct_probabilities")
    if cell_types is None:
        raise ValueError("No cell type probabilities found in experiment data")
    timings = experiment.images.timings()
    # Walk through all the time points, record once every _TIME_INTERVAL_H hours
    next_time_h = 0
    for time_point in experiment.positions.time_points():
        time_h = timings.get_time_h_since_start(time_point)
        if time_h < next_time_h:
            continue

        # Need to capture this time point
        next_time_h += _TIME_INTERVAL_H
        stem_to_paneth_locations = list()
        for position in experiment.positions.of_time_point(time_point):
            probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
            stem_to_paneth_location = cell_to_axis_function(cell_types, probabilities)
            if stem_to_paneth_location is not None:
                stem_to_paneth_locations.append(stem_to_paneth_location)

        into.add_values(time_h, stem_to_paneth_locations)


if __name__ == "__main__":
    main()
