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


class _CellTypeTransitionCounts:
    _cell_types: List[str]
    _transition_counts: ndarray

    def __init__(self, cell_types: List[str]):
        """Initializes the transition counts with the given cell types. You're allowed to pass an empty cell type list,
        in which case you'll get an instance that cannot store data. This is useful as a zero value in summing up
        instances of this class."""
        self._cell_types = cell_types
        self._transition_counts = numpy.zeros((len(cell_types), len(cell_types)), dtype=int)

    def add_transition(self, from_cell_type: str, to_cell_type: str):
        from_index = self._cell_types.index(from_cell_type)
        to_index = self._cell_types.index(to_cell_type)
        self._transition_counts[from_index, to_index] += 1

    def copy(self) -> "_CellTypeTransitionCounts":
        result = _CellTypeTransitionCounts(self._cell_types.copy())
        result._transition_counts = self._transition_counts.copy()
        return result

    def get_transition_count(self, from_cell_type: str, to_cell_type: str) -> int:
        """Gets the number of transitions from one cell type to another. Raises IndexError if the cell types are not
        in the list of cell types that this instance can store transition counts for."""
        from_index = self._cell_types.index(from_cell_type)
        to_index = self._cell_types.index(to_cell_type)
        return int(self._transition_counts[from_index, to_index])

    def cell_types(self) -> List[str]:
        """Gets the cell types that this instance can store transition counts for. Callers should not modify the
        returned list, as that would invalidate the data structure."""
        return self._cell_types

    def __add__(self, other: Any):
        if not isinstance(other, _CellTypeTransitionCounts):
            return NotImplemented

        if len(self._cell_types) == 0:
            return other.copy()
        if len(other._cell_types) == 0:
            return self.copy()
        if self._cell_types != other._cell_types:
            raise ValueError("Cannot add CellTypeTransitionCounts instances when both have specified different"
                             " cell types.")
        result = self.copy()
        result._transition_counts += other._transition_counts
        return result

    def transition_counts(self) -> ndarray:
        """Gets the transition count matrix. Modification should happen through self.add_transition()."""
        return self._transition_counts


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
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_REGENERATION, load_images=False):
        regeneration_data_by_experiment[experiment.name.get_name()] = _extract_transition_counts(experiment)

    # Collect control data
    control_data_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_CONTROL, load_images=False):
        control_data_by_experiment[experiment.name.get_name()] = _extract_transition_counts(experiment)

    # Sum up the data for each group
    regeneration_summed_data = sum(regeneration_data_by_experiment.values(), _CellTypeTransitionCounts([]))
    control_summed_data = sum(control_data_by_experiment.values(), _CellTypeTransitionCounts([]))

    print(control_summed_data._cell_types, control_summed_data._transition_counts, sep="\n")
    print(regeneration_summed_data._cell_types, regeneration_summed_data._transition_counts, sep="\n")

    _show_arrows_figure(control_summed_data, regeneration_summed_data)
    _show_heatmap_figure(control_summed_data, regeneration_summed_data)


def _show_arrows_figure(control_summed_data: _CellTypeTransitionCounts,
                        regeneration_summed_data: _CellTypeTransitionCounts):
    figure = lib_figures.new_figure(size=(5, 4))
    ax_control, ax_regen = figure.subplots(ncols=2, nrows=1)
    ax_control.set_title("Control")
    _plot_arrows(ax_control, control_summed_data)
    ax_regen.set_title("Regeneration")
    _plot_arrows(ax_regen, regeneration_summed_data)
    figure.tight_layout()
    plt.show()


def _plot_arrows(ax: Axes, transition_data: _CellTypeTransitionCounts):
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")

    # Add dots for the cell types
    cell_types = ["STEM", "MATURE_GOBLET", "ENTEROCYTE", "PANETH"]
    x_positions = [-1, 0, 1, 0]
    y_positions = [0, 1.7, 0, -1.7]
    ax.scatter(x_positions, y_positions, s=1000,
               c=[lib_figures.CELL_TYPE_PALETTE[cell_type] for cell_type in cell_types], zorder=2)

    transition_counts = [transition_data.get_transition_count("STEM", to_cell_type) for to_cell_type in cell_types]
    x_cell_type_from = x_positions[0]
    y_cell_type_from = y_positions[0]
    for i, count in enumerate(transition_counts):
        if i == 0:
            # Stem cell to stem cell transitions
            x = x_cell_type_from - 0.2
            y = y_cell_type_from + 0.75
            _circular_arrow(ax, 0.5, x, y, 310, 270, color="black")
            ax.text(x - 0.3, y, f"{count / sum(transition_counts) * 100:.1f}%",
                    ha="right", va="center", zorder=3)
            continue

        x_cell_type_to = x_positions[i]
        y_cell_type_to = y_positions[i]

        # Draw arrow from position 33% along the line to 66% along the line
        x_start = x_cell_type_from * 0.66 + x_cell_type_to * 0.33
        y_start = y_cell_type_from * 0.66 + y_cell_type_to * 0.33
        x_end = x_cell_type_from * 0.33 + x_cell_type_to * 0.66
        y_end = y_cell_type_from * 0.33 + y_cell_type_to * 0.66

        ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, head_width=0.1, head_length=0.1, fc='black',
                 ec='black', zorder=1)

        # Add percentage to the arrow
        ax.text((x_start + x_end) / 2, (y_start + y_end) / 2, f"{count / sum(transition_counts) * 100:.1f}%",
                ha="center", va="center", zorder=3)


def _circular_arrow(ax: Axes, diameter: float, cent_x: float, cent_y: float, start_angle: float, angle: float, **kwargs):
    startarrow = kwargs.pop("startarrow", False)
    endarrow = kwargs.pop("endarrow", False)

    arc = Arc([cent_x, cent_y], diameter, diameter, angle=start_angle,
              theta1=numpy.rad2deg(kwargs.get("head_length", 1.5 * 3 * .001)) if startarrow else 0,
              theta2=angle - (numpy.rad2deg(kwargs.get("head_length", 1.5 * 3 * .001)) if endarrow else 0),
              linestyle="-", color=kwargs.get("color", "black"))
    ax.add_patch(arc)

    if startarrow:
        startX = diameter / 2 * numpy.cos(numpy.radians(start_angle))
        startY = diameter / 2 * numpy.sin(numpy.radians(start_angle))
        startDX = +.000001 * diameter / 2 * numpy.sin(
            numpy.radians(start_angle) + kwargs.get("head_length", 1.5 * 3 * .001))
        startDY = -.000001 * diameter / 2 * numpy.cos(
            numpy.radians(start_angle) + kwargs.get("head_length", 1.5 * 3 * .001))
        ax.arrow(startX - startDX, startY - startDY, startDX, startDY, **kwargs)

    if endarrow:
        endX = diameter / 2 * numpy.cos(numpy.radians(start_angle + angle))
        endY = diameter / 2 * numpy.sin(numpy.radians(start_angle + angle))
        endDX = -.000001 * diameter / 2 * numpy.sin(
            numpy.radians(start_angle + angle) - kwargs.get("head_length", 1.5 * 3 * .001))
        endDY = +.000001 * diameter / 2 * numpy.cos(
            numpy.radians(start_angle + angle) - kwargs.get("head_length", 1.5 * 3 * .001))
        ax.arrow(endX - endDX, endY - endDY, endDX, endDY, **kwargs)


def _show_heatmap_figure(control_summed_data: _CellTypeTransitionCounts,
                         regeneration_summed_data: _CellTypeTransitionCounts):
    figure = lib_figures.new_figure()
    ax_control, ax_regen = figure.subplots(ncols=2, nrows=1)
    ax_control.set_title("Control")
    _plot_heatmap(ax_control, control_summed_data)
    ax_regen.set_title("Regeneration")
    _plot_heatmap(ax_regen, regeneration_summed_data)
    figure.tight_layout()
    plt.show()


def _plot_heatmap(ax: Axes, transition_data: _CellTypeTransitionCounts):
    transition_counts = transition_data.transition_counts()
    ax.imshow(transition_counts, cmap='gray', interpolation='nearest')
    ax.set_xticks(numpy.arange(len(transition_data.cell_types())))
    ax.set_yticks(numpy.arange(len(transition_data.cell_types())))
    ax.set_xticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in transition_data.cell_types()],
                       rotation=-45, ha="right")
    ax.set_yticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in transition_data.cell_types()])
    ax.set_xlabel("To")
    ax.set_ylabel("From")

    # Add counts to the cells
    max_count = numpy.max(transition_counts)
    for i in range(len(transition_data.cell_types())):
        for j in range(len(transition_data.cell_types())):
            ax.text(j, i, str(transition_counts[i, j]), ha="center", va="center",
                    color="black" if transition_counts[i, j] > max_count / 2 else "white")


def _extract_transition_counts(experiment: Experiment) -> _CellTypeTransitionCounts:
    cell_types = experiment.global_data.get_data("ct_probabilities")
    experiment_data = _CellTypeTransitionCounts(cell_types)

    for track in experiment.links.find_all_tracks():
        if not track.will_divide():
            continue

        cell_type_mother = _find_cell_type(experiment.position_data, track)
        if cell_type_mother is None:
            continue

        for daughter_track in track.get_next_tracks():
            cell_type_daughter = _find_cell_type(experiment.position_data, daughter_track)
            if cell_type_daughter is None:
                continue
            experiment_data.add_transition(cell_type_mother, cell_type_daughter)

    return experiment_data


if __name__ == "__main__":
    main()
