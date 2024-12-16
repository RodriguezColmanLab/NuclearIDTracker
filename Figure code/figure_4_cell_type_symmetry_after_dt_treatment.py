from typing import Optional, List, Union, Any

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.patches import Arc
from numpy import ndarray

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_AVERAGING_WINDOW_WIDTH_H = 5
_DATASET_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


class _CellTypeSymmetryCounts:
    _cell_types: List[str]
    _sister_cell_type_counts: ndarray  # Triangular matrix, with the cell types of the sisters as rows and columns

    def __init__(self, cell_types: List[str]):
        """Initializes the transition counts with the given cell types. You're allowed to pass an empty cell type list,
        in which case you'll get an instance that cannot store data. This is useful as a zero value in summing up
        instances of this class."""
        self._cell_types = cell_types
        self._sister_cell_type_counts = numpy.zeros((len(cell_types), len(cell_types)), dtype=numpy.float32)
        for i in range(len(cell_types)):
            for j in range(len(cell_types)):
                if j > i:
                    self._sister_cell_type_counts[i, j] = numpy.nan

    def add_sister_pair(self, sister_one_type: str, sister_two_type: str):
        sister_one_index = self._cell_types.index(sister_one_type)
        sister_two_index = self._cell_types.index(sister_two_type)

        if sister_two_index < sister_one_index:
            # This makes sure the matrix is triangular: first index must always be lower than or equal to the second
            sister_one_index, sister_two_index = sister_two_index, sister_one_index

        self._sister_cell_type_counts[sister_one_index, sister_two_index] += 1

    def copy(self) -> "_CellTypeSymmetryCounts":
        result = _CellTypeSymmetryCounts(self._cell_types.copy())
        result._sister_cell_type_counts = self._sister_cell_type_counts.copy()
        return result

    def __add__(self, other: Any):
        if not isinstance(other, _CellTypeSymmetryCounts):
            return NotImplemented

        if len(self._cell_types) == 0:
            return other.copy()
        if len(other._cell_types) == 0:
            return self.copy()
        if self._cell_types != other._cell_types:
            raise ValueError("Cannot add CellTypeTransitionCounts instances when both have specified different"
                             " cell types.")
        result = self.copy()
        result._sister_cell_type_counts += other._sister_cell_type_counts
        return result

    def sister_cell_type_counts(self) -> ndarray:
        return self._sister_cell_type_counts

    def cell_types(self) -> List[str]:
        return self._cell_types


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
        regeneration_data_by_experiment[experiment.name.get_name()] = _extract_sister_fates(experiment)

    # Collect control data
    control_data_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_CONTROL):
        control_data_by_experiment[experiment.name.get_name()] = _extract_sister_fates(experiment)

    # Sum up the data for each group
    regeneration_summed_data = sum(regeneration_data_by_experiment.values(), _CellTypeSymmetryCounts([]))
    control_summed_data = sum(control_data_by_experiment.values(), _CellTypeSymmetryCounts([]))

    print(control_summed_data.cell_types(), control_summed_data.sister_cell_type_counts(), sep="\n")
    print(regeneration_summed_data.cell_types(), regeneration_summed_data.sister_cell_type_counts(), sep="\n")

    _show_heatmap_figure(control_summed_data, regeneration_summed_data)


def _show_heatmap_figure(control_summed_data: _CellTypeSymmetryCounts,
                         regeneration_summed_data: _CellTypeSymmetryCounts):
    figure = lib_figures.new_figure()
    ax_control, ax_regen = figure.subplots(ncols=2, nrows=1)
    ax_control.set_title("Control")
    _plot_heatmap(ax_control, control_summed_data)
    ax_regen.set_title("Regeneration")
    _plot_heatmap(ax_regen, regeneration_summed_data)
    figure.tight_layout()
    plt.show()


def _plot_heatmap(ax: Axes, transition_data: _CellTypeSymmetryCounts):
    sister_counts = transition_data.sister_cell_type_counts()
    ax.imshow(sister_counts, cmap='magma', interpolation='nearest')
    ax.set_xticks(numpy.arange(len(transition_data.cell_types())))
    ax.set_yticks(numpy.arange(len(transition_data.cell_types())))
    ax.set_xticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in transition_data.cell_types()],
                       rotation=-45, ha="right")
    ax.set_yticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in transition_data.cell_types()])
    ax.set_xlabel("Sister 2")
    ax.set_ylabel("Sister 1")

    # Add counts to the cells
    max_count = numpy.nanmax(sister_counts)
    for i in range(len(transition_data.cell_types())):
        for j in range(len(transition_data.cell_types())):
            if j > i:
                continue
            ax.text(j, i, str(int(sister_counts[i, j])), ha="center", va="center",
                    color="black" if sister_counts[i, j] > max_count / 2 else "white")


def _extract_sister_fates(experiment: Experiment) -> _CellTypeSymmetryCounts:
    cell_types = experiment.global_data.get_data("ct_probabilities")
    experiment_data = _CellTypeSymmetryCounts(cell_types)

    for track in experiment.links.find_all_tracks():
        daughter_tracks = track.get_next_tracks()
        if len(daughter_tracks) != 2:
            continue

        daughter_1, daughter_2 = daughter_tracks
        cell_type_1 = _find_cell_type(experiment.position_data, daughter_1)
        cell_type_2 = _find_cell_type(experiment.position_data, daughter_2)
        if cell_type_1 is None or cell_type_2 is None:
            continue

        experiment_data.add_sister_pair(cell_type_1, cell_type_2)

    return experiment_data


if __name__ == "__main__":
    main()
