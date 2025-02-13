from typing import List, Optional, Iterable

import matplotlib
import numpy
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io

import lib_figures

_DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"
_TIME_INTERVAL_H = 5


class _StemToEnterocyteData:
    """Stores the stemness of cells along the stem-to-enterocyte axis, for multiple time points."""
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


def _find_stem_to_ec_location(cell_types: List[str], position_data: PositionData, position: Position) -> Optional[float]:
    """Projects a cell on the stem-to-enterocyte axis. If a cell has no predicted type, or a type other than stem or
    enterocyte, None is returned."""
    stemness = 0
    ct_probabilities = position_data.get_position_data(position, "ct_probabilities")
    if ct_probabilities is None:
        return None

    highest_type = cell_types[numpy.argmax(ct_probabilities)]
    if highest_type not in {"STEM", "ENTEROCYTE"}:
        return None  # Only consider stem and enterocyte cells

    for i, cell_type in enumerate(cell_types):
        if cell_type == "STEM":
            stemness += ct_probabilities[i]
        elif cell_type == "ENTEROCYTE":
            continue
        else:
            stemness = ct_probabilities[i] / 2  # Divide the remainder between stemness and enterocyteness
    return stemness


def main():

    all_probabilities = _StemToEnterocyteData()

    for experiment in list_io.load_experiment_list_file(_DATA_FILE, load_images=False):
        _collect_experiment_data(experiment, into=all_probabilities)

    figure = lib_figures.new_figure()
    ax = figure.gca()

    # Add p-values
    times_h = all_probabilities.times_h()
    for i in range(len(times_h) - 1):
        values = all_probabilities.values()[i]
        next_values = all_probabilities.values()[i + 1]
        t_statistic, p_value = ttest_ind(values, next_values)

        ax.text((times_h[i] + times_h[i + 1]) / 2, 0.5, f"p={p_value:.2f}", fontsize=8, ha="center", va="center")


    ax.violinplot(all_probabilities.values(), positions=all_probabilities.times_h(), showmeans=False, showmedians=True, showextrema=False, widths=8)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Stemness")
    ax.set_title("Stemness of cells along the stem-enterocyte axis")
    ax.set_xticks(all_probabilities.times_h())
    plt.show()



def _collect_experiment_data(experiment: Experiment, *, into: _StemToEnterocyteData):
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
        stem_to_ec_locations = list()
        for position in experiment.positions.of_time_point(time_point):
            stem_to_ec_location = _find_stem_to_ec_location(cell_types, experiment.position_data, position)
            if stem_to_ec_location is not None:
                stem_to_ec_locations.append(stem_to_ec_location)

        into.add_values(time_h, stem_to_ec_locations)


if __name__ == "__main__":
    main()