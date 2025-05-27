from collections import defaultdict

from typing import Optional, List, Union, Any, Dict

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.patches import Arc
from numpy import ndarray

import lib_data

import lib_figures
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATASET_FILE_ABLATION = "../../Data/Stem cell regeneration/Dataset - during DT treatment.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_X_LIM = (0.1, 0.79)


def _find_cell_type(position_data: PositionData, track: LinkingTrack) -> Optional[str]:
    """Finds the most common cell type in the track. Returns None if no cell type is found at all in the track."""
    cell_type_counts = dict()
    for position in track.positions():
        cell_type = position_markers.get_position_type(position_data, position)
        if cell_type is not None:
            cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
    if len(cell_type_counts) == 0:
        return None
    return max(cell_type_counts, key=cell_type_counts.get)


def _plot_organoid_averages(ax: Axes, data: Dict[str, List[float]], x: float):
    means = [numpy.mean(data_points) for data_points in data.values()]
    random = numpy.random.Generator(numpy.random.MT19937(seed=int(x)))
    x_positions = random.normal(x, 0.06, size=len(means))
    ax.scatter(x_positions, means, color="black", s=10, marker="s", lw=0)


def main():
    # Collect ablation data
    stem_to_ec_during_ablation = defaultdict(list)
    experiments_by_time_point = defaultdict(lambda: 0)
    ablation_timings = None
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_ABLATION):
        for time_point in experiment.positions.time_points():
            stem_to_ec_during_ablation[time_point] += _extract_stem_to_ec_locations(experiment, at_time_point=time_point)
            experiments_by_time_point[time_point] += 1
        ablation_timings = experiment.images.timings()

    # Remove time points in which not all experiments have data
    max_number_of_experiments = max(experiments_by_time_point.values())
    for time_point in list(stem_to_ec_during_ablation.keys()):
        if experiments_by_time_point[time_point] < max_number_of_experiments:
            del stem_to_ec_during_ablation[time_point]

    # Remove some selected time points
    del stem_to_ec_during_ablation[ablation_timings.find_closest_time_point(6, "h")]
    del stem_to_ec_during_ablation[ablation_timings.find_closest_time_point(10, "h")]

    # Collect regeneration data
    stem_to_ec_regeneration_start = list()
    stem_to_ec_regeneration_6h = list()
    stem_to_ec_regeneration_12h = list()
    stem_to_ec_regeneration_end = list()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_REGENERATION):
        stem_to_ec_regeneration_start += _extract_stem_to_ec_locations(experiment, at_time_point=experiment.positions.first_time_point())
        stem_to_ec_regeneration_6h += _extract_stem_to_ec_locations(experiment, at_time_point=experiment.images.timings().find_closest_time_point(6, "h"))
        stem_to_ec_regeneration_12h += _extract_stem_to_ec_locations(experiment, at_time_point=experiment.images.timings().find_closest_time_point(12, "h"))
        stem_to_ec_regeneration_end += _extract_stem_to_ec_locations(experiment, at_time_point=experiment.positions.last_time_point())

    times_names = ([f"Ablation {ablation_timings.get_time_h_since_start(time_point):.0f}h"
                   for time_point in stem_to_ec_during_ablation.keys()]
                   + ["Regeneration start", "+6h", "+12h", "+24h"])

    all_probabilities_values = (list(stem_to_ec_during_ablation.values()) +
                                [stem_to_ec_regeneration_start, stem_to_ec_regeneration_6h, stem_to_ec_regeneration_12h, stem_to_ec_regeneration_end])

    # Plot the data
    figure = lib_figures.new_figure()
    axes = figure.subplots(nrows=len(times_names), ncols=1, sharex=False, sharey=True)
    for i, ax in enumerate(axes):
        _, bins, patches = ax.hist(all_probabilities_values[i], bins=numpy.arange(0, 1, 0.02))
        for j in range(len(bins) - 1):
            bin_x = (bins[j] + bins[j + 1]) / 2
            patches[j].set_facecolor(lib_figures.get_stem_to_ec_color(bin_x))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i != len(times_names) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
        else:
            ax.set_xticks(numpy.arange(_X_LIM[0], _X_LIM[1] + 0.1, 0.1))
            ax.set_xlabel("Stem to enterocyte axis")
        ax.set_xlim(_X_LIM[1], _X_LIM[0])

        ax.set_yscale("log")
        ax.set_ylim(8, 700)
        ax.set_yticks([10, 100])
        ax.set_yticklabels(["10", "100"])

        # Add time to top left of panel
        ax.text(1.0, 0.8, times_names[i], transform=ax.transAxes, horizontalalignment="right", verticalalignment="top")

    axes[len(axes) // 2].set_ylabel("Cell count")
    plt.show()


def _extract_stem_to_ec_locations(experiment: Experiment, *, at_time_point: TimePoint) -> List[float]:
    """Gets the probabilities for the given cell type of all predicted *STEM* cells."""
    cell_types = experiment.global_data.get_data("ct_probabilities")
    stem_to_ec_locations = list()

    for position in experiment.positions.of_time_point(at_time_point):

        ct_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
        stem_to_ec_location = lib_data.find_stem_to_ec_location(cell_types, ct_probabilities)
        if stem_to_ec_location is not None:
            stem_to_ec_locations.append(stem_to_ec_location)

    return stem_to_ec_locations


if __name__ == "__main__":
    main()
