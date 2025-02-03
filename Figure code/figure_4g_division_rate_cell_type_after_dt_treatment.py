from collections import defaultdict

import numpy.random
from typing import Optional, Dict, Iterable

from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_AVERAGING_WINDOW_WIDTH_H = 5
_DATASET_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


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
        ax.text(x_positions_bars[i], rate, str(divisions_seen[i]), ha="center", va="bottom")

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

    ax.set_ylabel("Division rate (divisions/day)")
    ax.set_xlabel("Predicted cell type")
    ax.set_xticks(x_positions_control_bars + x_positions_regeneration_bars)
    ax.set_xticklabels(["control"] * len(x_positions_control_bars) + ["regeneration"] * len(x_positions_regeneration_bars),
                       rotation=-45, ha="left")

    for x, cell_type in zip(x_positions_cell_types, cell_types):
        ax.text(x, -0.15, lib_figures.style_cell_type_name(cell_type), ha="center", va="bottom")

    figure.tight_layout()
    plt.show()


def _extract_division_rates(experiment: Experiment) -> _DivisionRates:
    experiment_data = _DivisionRates()
    timings = experiment.images.timings()
    for track in experiment.links.find_all_tracks():
        cell_type = _find_cell_type(experiment.position_data, track)
        if cell_type is None:
            continue

        data_for_cell_type = experiment_data[cell_type]
        track_duration_h = timings.get_time_h_since_start(
            track.last_time_point() + 1) - timings.get_time_h_since_start(track.first_time_point())

        data_for_cell_type.hours_seen += track_duration_h
        if track.will_divide():
            data_for_cell_type.divisions_seen += 1
    return experiment_data


if __name__ == "__main__":
    main()
