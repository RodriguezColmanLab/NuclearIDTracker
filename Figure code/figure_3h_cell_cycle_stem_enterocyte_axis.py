from collections import defaultdict
from typing import Optional, Dict, Iterable, List, Any

import numpy
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

import lib_figures
from organoid_tracker.core import MPLColor
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"
_BIN_COUNT = 20


class _SingleBinData:
    cell_cycle_times: List[float]

    def __init__(self):
        self.cell_cycle_times = []

    def __repr__(self):
        return f"CellTypeData(hours_seen={self.hours_seen}, divisions_seen={self.divisions_seen})"

    def __add__(self, other):
        if not isinstance(other, _SingleBinData):
            return NotImplemented
        result = _SingleBinData()
        result.cell_cycle_times += self.cell_cycle_times
        result.cell_cycle_times += other.cell_cycle_times
        return result


class _ExperimentCellCycleTimes:
    bins: List[_SingleBinData]

    def __init__(self):
        self.bins = [_SingleBinData() for _ in range(_BIN_COUNT)]

    def __add__(self, other):
        if not isinstance(other, _ExperimentCellCycleTimes):
            return NotImplemented
        result = _ExperimentCellCycleTimes()
        for i in range(len(self.bins)):
            result.bins[i] = self.bins[i] + other.bins[i]
        return result

    def add_entry(self, stem_to_ec_location: float, cell_cycle_time_h: float):
        bin_index = int(stem_to_ec_location * _BIN_COUNT)
        if bin_index == len(self.bins):
            bin_index -= 1  # For the edge case where stem_to_ec_location is exactly 1.0

        self.bins[bin_index].cell_cycle_times.append(cell_cycle_time_h)


def _color_violin(violin: Dict[str, Any], color: MPLColor, average_bar_color: MPLColor = "black"):
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(1)
    if "cmeans" in violin:
        violin["cmeans"].set_color(average_bar_color)
    if "cmedians" in violin:
        violin["cmedians"].set_color(average_bar_color)


def _find_stem_to_ec_location(experiment: Experiment, track: LinkingTrack) -> Optional[float]:
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

    # Discard all but stem and enterocyte
    cell_type = defined_cell_types[numpy.argmax(overall_probabilities)]
    if cell_type not in {"STEM", "ENTEROCYTE"}:
        return None

    stemness = float(overall_probabilities[defined_cell_types.index("STEM")])
    enterocyteness = float(overall_probabilities[defined_cell_types.index("ENTEROCYTE")])

    # Divide the remainder between stemness and enterocyteness
    remainder = 1 - stemness - enterocyteness
    stemness += remainder / 2
    enterocyteness += remainder / 2

    # Now stemness equals 1 - enterocyteness, so we have our scale from stem to enterocyte
    return enterocyteness


def main():
    experiments = list_io.load_experiment_list_file(_DATA_FILE, load_images=False)

    # Collect data by experiment
    data_by_experiment = dict()
    for experiment in experiments:
        data_by_experiment[experiment.name.get_name()] = _get_experiment_cell_cycles(experiment)
    summed_data = sum(data_by_experiment.values(), _ExperimentCellCycleTimes())

    # Make a swarm plot per bin
    figure = lib_figures.new_figure()
    ax = figure.gca()
    bin_indices = numpy.arange(len(summed_data.bins))

    cell_cycle_times = [summed_data.bins[i].cell_cycle_times for i in bin_indices]
    colors = [
        lib_figures.get_mixed_cell_type_color(["STEM", "ENTEROCYTE", "PANETH"], [1 - i / _BIN_COUNT, i / _BIN_COUNT, 0])
        for i in bin_indices]

    random = numpy.random.Generator(numpy.random.MT19937(seed=1))
    for i, cell_cycle_times_for_bin, color in zip(bin_indices, cell_cycle_times, colors):
        if len(cell_cycle_times_for_bin) > 3:
            _color_violin(ax.violinplot([cell_cycle_times_for_bin], positions=[i], widths=0.7, showextrema=False, showmeans=False,
                                        showmedians=True), color=color)
        if len(cell_cycle_times_for_bin) > 0:
            ax.scatter(random.normal(loc=i, scale=0.07, size=len(cell_cycle_times_for_bin)), cell_cycle_times_for_bin,
                       c="black", s=4, marker="s", alpha=0.6, linewidths=0, zorder=10)

        # Plot p-value compared to the bin on the left
        cell_cycle_times_for_bin_left = cell_cycle_times[i - 1] if i > 0 else []
        if len(cell_cycle_times_for_bin) > 0 and len(cell_cycle_times_for_bin_left) > 0:
            t_statistic, p_value = ttest_ind(cell_cycle_times_for_bin, cell_cycle_times_for_bin_left)
            ax.text(i - 0.5, max(cell_cycle_times_for_bin + cell_cycle_times_for_bin_left), f"p = {p_value:.2f}", ha="center", va="top")

    ax.set_ylabel("Cell cycle time (h)")
    ax.set_xlabel("Stem-to-enterocyte-axis")
    ax.set_xticks(bin_indices - 0.5)
    ax.set_xticklabels([(f"{1 - i / _BIN_COUNT:.1f}" if i % 2 == 0 else "") for i in bin_indices])

    # Find min and max plotting x
    min_i = _BIN_COUNT / 2
    max_i = _BIN_COUNT / 2
    max_cell_cycle_h = 1
    for bin_index in bin_indices:
        if len(cell_cycle_times[bin_index]) > 0:
            min_i = min(min_i, bin_index)
            max_i = max(max_i, bin_index)
            max_cell_cycle_h = max(max_cell_cycle_h, max(cell_cycle_times[bin_index]))

    ax.set_xlim(min_i - 1, max_i + 1)
    ax.set_ylim(0, max_cell_cycle_h * 1.1)

    figure.tight_layout()
    plt.show()


def _get_experiment_cell_cycles(experiment: Experiment) -> _ExperimentCellCycleTimes:
    """Collects the division rate data for an experiment, by stem-to-enterocyte likelihood."""
    experiment_data = _ExperimentCellCycleTimes()
    timings = experiment.images.timings()
    for track in experiment.links.find_all_tracks():
        stem_to_ec_location = _find_stem_to_ec_location(experiment, track)
        if stem_to_ec_location is None:
            continue

        previous_tracks = track.get_previous_tracks()
        if len(previous_tracks) != 1:
            continue
        if not previous_tracks.pop().will_divide():
            continue
        if not track.will_divide():
            continue

        # At this point, we know we captured a full cell cycle
        previous_division_time_point = track.first_time_point() - 1
        next_division_time_point = track.last_time_point()
        previous_division_time_h = timings.get_time_h_since_start(previous_division_time_point)
        next_division_time_h = timings.get_time_h_since_start(next_division_time_point)
        cell_cycle_time_h = next_division_time_h - previous_division_time_h

        experiment_data.add_entry(stem_to_ec_location, cell_cycle_time_h)
    return experiment_data


if __name__ == "__main__":
    main()
