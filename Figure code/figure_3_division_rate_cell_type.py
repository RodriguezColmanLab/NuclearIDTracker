from collections import defaultdict
from typing import Optional, Dict, Iterable

from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_AVERAGING_WINDOW_WIDTH_H = 5
_DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"


# Tracks shorter than this are ignored. Sometimes, cells move into the field of view to divide, and then are tracked
# for a very short time. At division, the stemness is very high, so then you get an artificially high division rate.
_MIN_TRACK_DURATION_H = 2


class _CellTypeData:
    hours_seen: float
    divisions_seen: int

    def __init__(self):
        self.hours_seen = 0
        self.divisions_seen = 0

    def __repr__(self):
        return f"CellTypeData(hours_seen={self.hours_seen}, divisions_seen={self.divisions_seen})"

    def __add__(self, other):
        if not isinstance(other, _CellTypeData):
            return NotImplemented
        result = _CellTypeData()
        result.hours_seen = self.hours_seen + other.hours_seen
        result.divisions_seen = self.divisions_seen + other.divisions_seen
        return result


class _ExperimentData:
    data_by_cell_type: Dict[str, _CellTypeData]

    def __init__(self):
        self.data_by_cell_type = defaultdict(_CellTypeData)

    def __repr__(self):
        return f"ExperimentData(data_by_cell_type={self.data_by_cell_type})"

    def __add__(self, other):
        if not isinstance(other, _ExperimentData):
            return NotImplemented
        result = _ExperimentData()
        for cell_type in self.data_by_cell_type.keys() | other.data_by_cell_type.keys():
            result.data_by_cell_type[cell_type] = self.data_by_cell_type[cell_type] + other.data_by_cell_type[cell_type]
        return result

    def keys(self) -> Iterable[str]:
        return self.data_by_cell_type.keys()

    def __getitem__(self, item: str) -> _CellTypeData:
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
    experiments = list_io.load_experiment_list_file(_DATA_FILE, load_images=False)

    # Collect data by cell type
    data_by_experiment = dict()
    for experiment in experiments:
        experiment_data = _ExperimentData()
        timings = experiment.images.timings()
        for track in experiment.links.find_all_tracks():
            cell_type = _find_cell_type(experiment.position_data, track)
            if cell_type is None:
                continue

            data_for_cell_type = experiment_data[cell_type]
            track_duration_h = timings.get_time_h_since_start(
                track.last_time_point() + 1) - timings.get_time_h_since_start(track.first_time_point())

            if track_duration_h < _MIN_TRACK_DURATION_H:
                continue

            data_for_cell_type.hours_seen += track_duration_h
            if track.will_divide():
                data_for_cell_type.divisions_seen += 1
        data_by_experiment[experiment.name.get_name()] = experiment_data

    summed_data = sum(data_by_experiment.values(), _ExperimentData())

    # Filter out cell types where we have less than 24h of footage
    summed_data = {cell_type: data for cell_type, data in summed_data.data_by_cell_type.items() if data.hours_seen > 24}

    # Sort by division rate
    summed_data = {cell_type: data for cell_type, data in sorted(summed_data.items(), reverse=True,
                                                                 key=lambda x: x[1].divisions_seen / x[1].hours_seen)}

    # Make a bar graph, showing the division rate for each cell type
    figure = lib_figures.new_figure()
    ax = figure.gca()
    cell_types = [cell_type for cell_type in summed_data.keys() if summed_data[cell_type].hours_seen > 10]

    hours_seen = [summed_data[cell_type].hours_seen for cell_type in cell_types]
    divisions_seen = [summed_data[cell_type].divisions_seen for cell_type in cell_types]
    division_rate_h = [divisions / hours for hours, divisions in zip(hours_seen, divisions_seen)]
    division_rate_day = [rate * 24 for rate in division_rate_h]
    ax.bar(list(range(len(cell_types))), division_rate_day, color=[lib_figures.CELL_TYPE_PALETTE[cell_type] for cell_type in cell_types])

    # Add number of divisions to the bars
    for i, rate in enumerate(division_rate_day):
        ax.text(i, rate, str(int(hours_seen[i] / 24)) + "d", ha="center", va="bottom")

    # Add spread of division rates per experiment
    for x, cell_type in enumerate(cell_types):
        hours_seen = [experiment_data[cell_type].hours_seen for experiment_data in data_by_experiment.values()]
        divisions_seen = [experiment_data[cell_type].divisions_seen for experiment_data in data_by_experiment.values()]
        division_rate_h = [divisions / hours if hours > 0 else 0 for hours, divisions in zip(hours_seen, divisions_seen)]
        division_rate_day = [rate * 24 for rate in division_rate_h]
        ax.scatter([x] * len(division_rate_day), division_rate_day, color="black", s=13, linewidth=1, edgecolor="white")

    ax.set_ylabel("Division rate (divisions / cell / day)")
    ax.set_xlabel("Predicted cell type")
    ax.set_xticks(list(range(len(cell_types))))
    ax.set_xticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in cell_types], rotation=-45,
                       ha="left")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
