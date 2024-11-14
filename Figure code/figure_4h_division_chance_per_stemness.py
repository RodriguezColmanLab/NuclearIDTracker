import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from typing import Optional, List

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.imaging import list_io

#_DATA_FILE = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_DATA_FILE = "../../Data/Predicted data.autlist"
_DIVISION_MEASUREMENT_TIME_H = 15


class _DivisionChancePerStemness:
    name: str
    stemness_values: ndarray
    dividing_cells_by_stemness: ndarray
    non_dividing_cells_by_stemness: ndarray

    def __init__(self, name: str):
        self.name = name
        self.stemness_values = numpy.linspace(0, 1, 11, endpoint=True)
        self.dividing_cells_by_stemness = numpy.zeros_like(self.stemness_values, dtype=int)
        self.non_dividing_cells_by_stemness = numpy.zeros_like(self.stemness_values, dtype=int)

    def add_cell(self, stemness: float, will_divide: bool):
        stemness_index = numpy.searchsorted(self.stemness_values, stemness) - 1
        if will_divide:
            self.dividing_cells_by_stemness[stemness_index] += 1
        else:
            self.non_dividing_cells_by_stemness[stemness_index] += 1


def _draw_plot(ax: Axes, division_chance_per_stemness: _DivisionChancePerStemness):
    ax.set_title(division_chance_per_stemness.name)
    bar_width = division_chance_per_stemness.stemness_values[1] - division_chance_per_stemness.stemness_values[0]
    sums = division_chance_per_stemness.dividing_cells_by_stemness + division_chance_per_stemness.non_dividing_cells_by_stemness

    sums_nonzero = sums.copy()
    sums_nonzero[sums_nonzero == 0] = 1
    fraction_dividing = division_chance_per_stemness.dividing_cells_by_stemness / sums_nonzero
    fraction_non_dividing = division_chance_per_stemness.non_dividing_cells_by_stemness / sums_nonzero

    for i in range(len(division_chance_per_stemness.stemness_values) - 1):
        x = division_chance_per_stemness.stemness_values[i] + bar_width / 2
        ax.text(x, 1, str(sums[i]), ha="center", va="bottom")

    ax.bar(division_chance_per_stemness.stemness_values, fraction_non_dividing, align="edge",
           color="#0984e3", label="Will not divide", width=bar_width, alpha=0.8)
    ax.bar(division_chance_per_stemness.stemness_values, fraction_dividing, align="edge",
           color="#d63031", label="Will divide", bottom=fraction_non_dividing,
           width=bar_width, alpha=0.8)
    ax.set_xlabel("Stemness")
    ax.set_ylabel("Number of cells")
    ax.set_xticks(division_chance_per_stemness.stemness_values)

    ax.legend()


def _seen_division_in_time(experiment: Experiment, track: LinkingTrack) -> bool:
    if not track.will_divide():
        return False

    # Check whether the division happens in the first _DIVISION_MEASUREMENT_TIME_H hours
    # Otherwise we ignore it
    division_time_point_number = track.last_time_point_number()
    division_time_h = experiment.images.timings().get_time_h_since_start(division_time_point_number)
    return division_time_h <= _DIVISION_MEASUREMENT_TIME_H


def main():
    plt.rcParams['savefig.dpi'] = 180

    experiments = list_io.load_experiment_list_file(_DATA_FILE, load_images=False)
    division_chances_per_stemness = []

    for experiment in experiments:
        division_chance_per_stemness = _DivisionChancePerStemness(experiment.name.get_name())
        for track in experiment.links.find_starting_tracks():
            if not _passes_filter(experiment, track):
                continue
            will_divide = _seen_division_in_time(experiment, track)
            stemness = _find_initial_stemness(experiment, track)
            if stemness is not None:
                division_chance_per_stemness.add_cell(stemness, will_divide)
        division_chances_per_stemness.append(division_chance_per_stemness)

    figure = lib_figures.new_figure()
    axes = figure.subplots(nrows=1, ncols=len(division_chances_per_stemness), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        _draw_plot(ax, division_chances_per_stemness[i])
    plt.show()


def _find_initial_stemness(experiment: Experiment, track: LinkingTrack) -> Optional[float]:
    cell_type_names = experiment.global_data.get_data("ct_probabilities")

    stem_probabilities = list()
    for i, position in enumerate(track.positions()):
        probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
        if probabilities is None:
            continue
        stem_probabilities.append(probabilities[cell_type_names.index("STEM")])
        if i > 5:
            break

    if len(stem_probabilities) == 0:
        return None
    return sum(stem_probabilities) / len(stem_probabilities)


def _passes_filter(experiment: Experiment, starting_track: LinkingTrack):
    if starting_track.first_time_point_number() > experiment.positions.first_time_point_number():
        return False

    last_time_point_number = _last_time_point_number(starting_track)
    last_time_h = experiment.images.timings().get_time_h_since_start(last_time_point_number)
    return last_time_h >= _DIVISION_MEASUREMENT_TIME_H


def _last_time_point_number(starting_track: LinkingTrack) -> int:
    last_time_point_number = starting_track.last_time_point_number()
    for track in starting_track.find_all_descending_tracks(include_self=False):
        last_time_point_number = max(last_time_point_number, track.last_time_point_number())
    return last_time_point_number


if __name__ == "__main__":
    main()
