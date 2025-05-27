"""We plot all stem and Paneth cells on a stem-to-Paneth axis, and see how their amount changes over time."""
from matplotlib.axes import Axes
from typing import List, Callable, Optional

import numpy
from matplotlib import pyplot as plt

import lib_data
import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.imaging import list_io

_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"

# Tracks shorter than this are ignored. Sometimes, cells move into the field of view to divide, and then are tracked
# for a very short time. At division, the stemness is very high, so then you get an artificially high division rate.
_MIN_TRACK_DURATION_H = 2


class _StemToXData:
    """Stores the stemness of cells along the stem-to-X (Paneth, enterocyte) axis, for multiple time points."""
    _axis_locations: List[float]

    def __init__(self):
        self._axis_locations = list()

    def add_value(self, value: float):
        self._axis_locations.append(value)

    def axis_locations(self) -> List[float]:
        return self._axis_locations


def main():
    stem_to_paneth_data_control = _StemToXData()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_CONTROL, load_images=False):
        _collect_experiment_data(experiment, into=stem_to_paneth_data_control)

    stem_to_paneth_data_regeneration = _StemToXData()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_REGENERATION, load_images=False):
        _collect_experiment_data(experiment, into=stem_to_paneth_data_regeneration)

    figure = lib_figures.new_figure()
    ax_control, ax_regeneration = figure.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    random = numpy.random.Generator(numpy.random.MT19937(seed=1))
    _plot_dots(ax_control, random, stem_to_paneth_data_control)

    ax_control.text(0.51, 0.35, "← Crypt", ha="right", va="bottom")
    ax_control.text(0.49, 0.35, "Paneth →", ha="left", va="bottom")

    for ax in [ax_control, ax_regeneration]:
        ax.set_xlim(0.8, 0.2)
        ax.set_ylim(-0.35, 0.42)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axvline(0.5, color="gray", linewidth=1.5)

    ax_regeneration.set_xlabel("Stem-to-Paneth axis")
    ax_control.set_title("Dividing cells in de-novo growth")
    ax_regeneration.set_title("Dividing cells in regeneration")

    _plot_dots(ax_regeneration, random, stem_to_paneth_data_regeneration)

    plt.show()


def _plot_dots(ax: Axes, random: numpy.random.Generator, data: _StemToXData):
    all_stem_to_paneth_values = data.axis_locations()
    y_values = random.normal(loc=0, scale=0.1, size=len(all_stem_to_paneth_values))
    colors = [lib_figures.get_stem_to_paneth_color(value) for value in all_stem_to_paneth_values]
    ax.scatter(all_stem_to_paneth_values, y_values, alpha=0.5, color=colors, s=10, linewidths=0.5,
               edgecolors="#2d3436")


def _find_stem_to_paneth_location(experiment: Experiment, track: LinkingTrack) -> Optional[float]:
    """Finds the point the cell lies on the stem-to-enterocyte axis. If a cell has no predicted type, or a type
    other than stem or enterocyte, None is returned."""
    defined_cell_types = experiment.global_data.get_data("ct_probabilities")
    if defined_cell_types is None:
        raise ValueError("No cell type probabilities found in experiment data")

    position_data = experiment.position_data
    overall_probabilities = numpy.zeros(len(defined_cell_types), dtype=float)
    included_position_count = 0

    for position in track.positions():
        probabilities = position_data.get_position_data(position, "ct_probabilities")
        if probabilities is None:
            continue
        overall_probabilities += probabilities
        included_position_count += 1

    if included_position_count == 0:
        return None
    overall_probabilities /= included_position_count  # Convert to average probabilities

    return lib_data.find_stem_to_paneth_location(defined_cell_types, overall_probabilities)


def _collect_experiment_data(experiment: Experiment, *, into: _StemToXData):
    timings = experiment.images.timings()

    for track in experiment.links.find_all_tracks():
        if not track.will_divide():
            continue

        track_duration_h = timings.get_time_h_since_start(
            track.last_time_point() + 1) - timings.get_time_h_since_start(track.first_time_point())

        if track_duration_h < _MIN_TRACK_DURATION_H:
            continue

        cell_type_axis_location = _find_stem_to_paneth_location(experiment, track)
        if cell_type_axis_location is None:
            continue
        into.add_value(cell_type_axis_location)


if __name__ == "__main__":
    main()
