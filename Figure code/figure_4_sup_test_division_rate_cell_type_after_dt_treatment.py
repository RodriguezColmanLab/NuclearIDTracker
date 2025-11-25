from collections import defaultdict

import numpy.random
from numpy import ndarray
from typing import Optional, Dict, Iterable, List

from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers
import lib_data

_AVERAGING_WINDOW_WIDTH_H = 5
_DATASET_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


# Tracks shorter than this are ignored. Sometimes, cells move into the field of view to divide, and then are tracked
# for a very short time. At division, the stemness is very high, so then you get an artificially high division rate.
_MIN_TRACK_DURATION_H = 2


class _CellTypeDivisionRate:
    hours_seen: float
    divisions_seen: int

    def __init__(self):
        self.hours_seen = 0
        self.divisions_seen = 0

    def __repr__(self):
        return f"CellTypeData(hours_seen={self.hours_seen}, divisions_seen={self.divisions_seen})"

    def __add__(self, other):
        if not isinstance(other, _CellTypeDivisionRate):
            return NotImplemented
        result = _CellTypeDivisionRate()
        result.hours_seen = self.hours_seen + other.hours_seen
        result.divisions_seen = self.divisions_seen + other.divisions_seen
        return result


class _DivisionRates:
    data_by_cell_type: Dict[str, _CellTypeDivisionRate]

    def __init__(self):
        self.data_by_cell_type = defaultdict(_CellTypeDivisionRate)

    def __repr__(self):
        return f"ExperimentData(data_by_cell_type={self.data_by_cell_type})"

    def __add__(self, other):
        if not isinstance(other, _DivisionRates):
            return NotImplemented
        result = _DivisionRates()
        for cell_type in self.data_by_cell_type.keys() | other.data_by_cell_type.keys():
            result.data_by_cell_type[cell_type] = self.data_by_cell_type[cell_type] + other.data_by_cell_type[cell_type]
        return result

    def keys(self) -> Iterable[str]:
        return self.data_by_cell_type.keys()

    def __getitem__(self, item: str) -> _CellTypeDivisionRate:
        return self.data_by_cell_type[item]


def _find_cell_type(position_data: PositionData, track: LinkingTrack, cell_types: List[str]) -> Optional[str]:
    """Finds the most common cell type in the track. Returns None if no cell type is found at all in the track."""
    cell_type_probabilities_sum: Optional[ndarray] = None
    cell_type_probabilities_count = 0

    for position in track.positions():
        # Record cell type probabilities
        cell_type_probabilities = position_data.get_position_data(position, "ct_probabilities")
        if cell_type_probabilities is None:
            continue

        if cell_type_probabilities_sum is None:
            cell_type_probabilities_sum = numpy.zeros(len(cell_type_probabilities), dtype=numpy.float32)

        cell_type_probabilities_sum += cell_type_probabilities
        cell_type_probabilities_count += 1

    # Calculate average
    if cell_type_probabilities_sum is None:
        return None

    cell_type_probabilities = cell_type_probabilities_sum / cell_type_probabilities_count

    if numpy.argmax(cell_type_probabilities) == cell_types.index("PANETH"):
        return "PANETH"

    stem_to_ec_loc = lib_data.find_stem_to_ec_location(cell_types, cell_type_probabilities)
    if stem_to_ec_loc is None:
        return None
    if stem_to_ec_loc > 0.55:
        return "STEM"
    if stem_to_ec_loc < 0.4:
        return "ENTEROCYTE"
    return "TA"


def main():
    # Collect regeneration data
    regeneration_data_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_REGENERATION):
        regeneration_data_by_experiment[experiment.name.get_name()] = _extract_division_rates(experiment)

    # Collect control data
    control_data_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_CONTROL):
        control_data_by_experiment[experiment.name.get_name()] = _extract_division_rates(experiment)

    # Sum up the data for each group
    regeneration_summed_data = sum(regeneration_data_by_experiment.values(), _DivisionRates())
    control_summed_data = sum(control_data_by_experiment.values(), _DivisionRates())

    # Make a bar graph, showing the division rate for each cell type
    figure = lib_figures.new_figure()
    ax = figure.gca()
    cell_types = [cell_type for cell_type in regeneration_summed_data.keys() if regeneration_summed_data[cell_type].hours_seen > 10]
    cell_types.sort(reverse=True)

    hours_seen = list()
    for cell_type in cell_types:
        hours_seen.append(control_summed_data[cell_type].hours_seen)
        hours_seen.append(regeneration_summed_data[cell_type].hours_seen)

    divisions_seen = list()
    for cell_type in cell_types:
        divisions_seen.append(control_summed_data[cell_type].divisions_seen)
        divisions_seen.append(regeneration_summed_data[cell_type].divisions_seen)

    division_rate_h = list()
    for cell_type in cell_types:
        division_rate_h.append(control_summed_data[cell_type].divisions_seen / control_summed_data[cell_type].hours_seen)
        division_rate_h.append(regeneration_summed_data[cell_type].divisions_seen / regeneration_summed_data[cell_type].hours_seen)
    division_rate_day = [rate * 24 for rate in division_rate_h]

    bar_colors = list()
    for cell_type in cell_types:
        bar_colors.append(lib_figures.CELL_TYPE_PALETTE[cell_type])  # Once for control
        bar_colors.append(lib_figures.CELL_TYPE_PALETTE[cell_type])  # and once for regeneration

    x_positions_bars = list()
    x_positions_control_bars = list()
    x_positions_regeneration_bars = list()
    x_positions_cell_types = list()
    for i, cell_type in enumerate(cell_types):
        x_positions_bars.append(i * 2)
        x_positions_control_bars.append(i * 2)
        x_positions_bars.append(i * 2 + 0.8)
        x_positions_regeneration_bars.append(i * 2 + 0.8)
        x_positions_cell_types.append(i * 2 + 0.4)  # Average position of the two bars

    ax.bar(x_positions_bars, division_rate_day, color=bar_colors)

    # Add number of divisions to the bars
    for i, rate in enumerate(division_rate_day):
        ax.text(x_positions_bars[i], rate, str(int(divisions_seen[i])), ha="center", va="bottom")

    # Add spread of division rates for the control experiment
    random_generator = numpy.random.Generator(numpy.random.MT19937(seed=1))
    for i, cell_type in enumerate(cell_types):
        hours_seen = [experiment_data[cell_type].hours_seen for experiment_data in control_data_by_experiment.values()]
        divisions_seen = [experiment_data[cell_type].divisions_seen for experiment_data in control_data_by_experiment.values()]
        division_rate_h = [divisions / hours if hours > 0 else 0 for hours, divisions in zip(hours_seen, divisions_seen)]
        division_rate_day = [rate * 24 for rate in division_rate_h]
        x_positions_scatter_points = random_generator.normal(x_positions_control_bars[i], 0.05, size=len(division_rate_day))
        ax.scatter(x_positions_scatter_points, division_rate_day, color="black", s=13, linewidth=1, edgecolor="white")

    # Add spread of division rates for the regeneration experiment
    for i, cell_type in enumerate(cell_types):
        hours_seen = [experiment_data[cell_type].hours_seen for experiment_data in regeneration_data_by_experiment.values()]
        divisions_seen = [experiment_data[cell_type].divisions_seen for experiment_data in regeneration_data_by_experiment.values()]
        division_rate_h = [divisions / hours if hours > 0 else 0 for hours, divisions in zip(hours_seen, divisions_seen)]
        division_rate_day = [rate * 24 for rate in division_rate_h]
        x_positions_scatter_points = random_generator.normal(x_positions_regeneration_bars[i], 0.05, size=len(division_rate_day))
        ax.scatter(x_positions_scatter_points, division_rate_day, color="black", s=13, linewidth=1, edgecolor="white")

    ax.set_ylabel("Division rate (divisions/cell/day)")
    ax.set_xlabel("Predicted cell type")
    ax.set_xticks(x_positions_control_bars + x_positions_regeneration_bars)
    ax.set_xticklabels(["control"] * len(x_positions_control_bars) + ["regeneration"] * len(x_positions_regeneration_bars),
                       rotation=-45, ha="left")

    for x, cell_type in zip(x_positions_cell_types, cell_types):
        ax.text(x, -0.15, cell_type.lower(), ha="center", va="bottom")

    figure.tight_layout()
    plt.show()


def _extract_division_rates(experiment: Experiment) -> _DivisionRates:
    experiment_data = _DivisionRates()
    timings = experiment.images.timings()
    cell_types = experiment.global_data.get_data("ct_probabilities")
    for track in experiment.links.find_all_tracks():
        cell_type = _find_cell_type(experiment.position_data, track, cell_types)
        if cell_type is None:
            continue

        data_for_cell_type = experiment_data[cell_type]
        track_duration_h = timings.get_time_h_since_start(
            track.last_time_point() + 1) - timings.get_time_h_since_start(track.first_time_point())

        if track_duration_h < _MIN_TRACK_DURATION_H:
            continue

        data_for_cell_type.hours_seen += track_duration_h
        if track.will_divide():
            if cell_type == "PANETH":
                stem_to_paneth_location = _find_stem_to_paneth_location(experiment, track)
                print(f"Found a Paneth cell that will divide in {experiment.name.get_name()}, location {stem_to_paneth_location}")
            data_for_cell_type.divisions_seen += 1
    return experiment_data


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


if __name__ == "__main__":
    main()
