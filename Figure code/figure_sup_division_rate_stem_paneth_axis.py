from typing import Optional, List

import numpy
from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.imaging import list_io

_DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"
_BIN_COUNT = 20

# Tracks shorter than this are ignored. Sometimes, cells move into the field of view to divide, and then are tracked
# for a very short time. At division, the stemness is very high, so then you get an artificially high division rate.
_MIN_TRACK_DURATION_H = 2

# Bins with less than this amount of footage are ignored. If we have such a small amount of footage, the division rate
# is very uncertain.
_MIN_BIN_FOOTAGE_H = 10


class _SingleBinData:
    hours_seen: float
    divisions_seen: int
    cells_seen: int

    def __init__(self):
        self.hours_seen = 0
        self.divisions_seen = 0
        self.cells_seen = 0

    def __repr__(self):
        return f"CellTypeData(hours_seen={self.hours_seen}, divisions_seen={self.divisions_seen}, cells_seen={self.cells_seen})"

    def __add__(self, other):
        if not isinstance(other, _SingleBinData):
            return NotImplemented
        result = _SingleBinData()
        result.hours_seen = self.hours_seen + other.hours_seen
        result.divisions_seen = self.divisions_seen + other.divisions_seen
        result.cells_seen = self.cells_seen + other.cells_seen
        return result


class _ExperimentData:
    bins: List[_SingleBinData]

    def __init__(self):
        self.bins = [_SingleBinData() for _ in range(_BIN_COUNT)]

    def __add__(self, other):
        if not isinstance(other, _ExperimentData):
            return NotImplemented
        result = _ExperimentData()
        for i in range(len(self.bins)):
            result.bins[i] = self.bins[i] + other.bins[i]
        return result

    def add_entry(self, stem_to_ec_location: float, hours: float, *, division_at_end: bool, count_cell: bool):
        """Adds an entry to the appropriate bin.

        count_cell is normally True, but should be set to False if the cell is dividing and both daughter cells are
        included in the analysis. Otherwise, a single dividing cell would be counted as 3 cells (mother + 2 daughters).
        """
        bin_index = int(stem_to_ec_location * _BIN_COUNT)
        if bin_index == len(self.bins):
            bin_index -= 1  # For the edge case where stem_to_ec_location is exactly 1.0

        self.bins[bin_index].hours_seen += hours
        if division_at_end:
            self.bins[bin_index].divisions_seen += 1
        if count_cell:
            self.bins[bin_index].cells_seen += 1


def _find_stem_to_paneth_location(experiment: Experiment, track: LinkingTrack) -> Optional[float]:
    """Finds the point the cell lies on the stem-to-Paneth axis. If a cell has no predicted type, or a type
    other than stem or Paneth, None is returned."""
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

    # Discard all but stem and Paneth
    cell_type = defined_cell_types[numpy.argmax(overall_probabilities)]
    if cell_type not in {"STEM", "PANETH"}:
        return None

    stemness = float(overall_probabilities[defined_cell_types.index("STEM")])
    panethness = float(overall_probabilities[defined_cell_types.index("PANETH")])

    # Divide the remainder between stemness and panethness
    remainder = 1 - stemness - panethness
    stemness += remainder / 2
    panethness += remainder / 2

    # Now stemness equals 1 - panethness, so we have our scale from stem to Paneth
    return panethness


def main():
    experiments = list_io.load_experiment_list_file(_DATA_FILE, load_images=False)

    # Collect data by experiment
    data_by_experiment = dict()
    for experiment in experiments:
        data_by_experiment[experiment.name.get_name()] = _get_experiment_division_rate(experiment)
    summed_data = sum(data_by_experiment.values(), _ExperimentData())

    # Make a bar graph, showing the division rate for each cell type
    figure = lib_figures.new_figure()
    ax = figure.gca()
    bin_indices = numpy.arange(len(summed_data.bins))

    hours_seen = [summed_data.bins[i].hours_seen for i in bin_indices]
    divisions_seen = [summed_data.bins[i].divisions_seen for i in bin_indices]
    division_rate_h = [(divisions / hours if hours > 0 else 0) for hours, divisions in zip(hours_seen, divisions_seen)]
    division_rate_day = [rate * 24 for rate in division_rate_h]
    colors = [
        lib_figures.get_stem_to_paneth_color(1 - i / _BIN_COUNT)
        for i in bin_indices]
    ax.bar(bin_indices, division_rate_day, width=1, color=colors)

    # Add number of divisions to the bars
    for bin_index, rate in enumerate(division_rate_day):
        hours_seen_at_index = hours_seen[bin_index]
        if hours_seen_at_index > _MIN_BIN_FOOTAGE_H:
            cells_seen = str(summed_data.bins[bin_index].cells_seen)
            ax.text(bin_index, rate, cells_seen, ha="center", va="bottom")

    # Add spread of division rates per experiment
    for bin_index in bin_indices:
        hours_seen_at_index = [experiment_data.bins[bin_index].hours_seen for experiment_data in
                               data_by_experiment.values()]
        if sum(hours_seen_at_index) < _MIN_BIN_FOOTAGE_H:
            continue  # This bin is not drawn

        divisions_seen = [experiment_data.bins[bin_index].divisions_seen for experiment_data in
                          data_by_experiment.values()]
        division_rate_h = [divisions / hours for hours, divisions in zip(hours_seen_at_index, divisions_seen) if
                           hours > _MIN_BIN_FOOTAGE_H]
        division_rate_day = [rate * 24 for rate in division_rate_h]
        ax.scatter([bin_index] * len(division_rate_day), division_rate_day, color="black", s=13, linewidth=1,
                   edgecolor="white")

    ax.set_ylabel("Division rate (divisions / cell / day)")
    ax.set_xlabel("Stem-to-Paneth-axis")
    ax.set_xticks(bin_indices - 0.5)
    ax.set_xticklabels([(f"{1 - i / _BIN_COUNT:.1f}" if i % 2 == 0 else "") for i in bin_indices])

    # Find min and max plotting x
    min_i = _BIN_COUNT / 2
    max_i = _BIN_COUNT / 2
    for bin_index in bin_indices:
        if hours_seen[bin_index] > _MIN_BIN_FOOTAGE_H:
            min_i = min(min_i, bin_index)
            max_i = max(max_i, bin_index)
    ax.set_xlim(min_i - 1, max_i + 1)

    figure.tight_layout()
    plt.show()


def _get_experiment_division_rate(experiment: Experiment) -> _ExperimentData:
    """Collects the division rate data for an experiment, by stem-to-Paneth likelihood."""
    experiment_data = _ExperimentData()
    timings = experiment.images.timings()
    for track in experiment.links.find_all_tracks():
        stem_to_ec_location = _find_stem_to_paneth_location(experiment, track)
        if stem_to_ec_location is None:
            continue

        track_duration_h = timings.get_time_h_since_start(
            track.last_time_point() + 1) - timings.get_time_h_since_start(track.first_time_point())

        if track_duration_h < _MIN_TRACK_DURATION_H:
            continue

        # If a cell divides into two, and both daughters are included in the analysis, we do not count the mother cell.
        # Otherwise, a single dividing cell would be counted as 3 cells (mother + 2 daughters), instead of 2 cells.
        will_divide = track.will_divide()
        daughters_included_for_division_rate = True
        for daughter_track in track.get_next_tracks():
            if _find_stem_to_paneth_location(experiment, daughter_track) is None or timings.get_time_h_since_start(
                    track.last_time_point() + 1) - timings.get_time_h_since_start(
                track.first_time_point()) < _MIN_TRACK_DURATION_H:
                daughters_included_for_division_rate = False
                break

        count_cell = True
        if will_divide and daughters_included_for_division_rate:
            count_cell = False

        experiment_data.add_entry(stem_to_ec_location, track_duration_h, division_at_end=will_divide,
                                  count_cell=count_cell)
    return experiment_data


if __name__ == "__main__":
    main()
