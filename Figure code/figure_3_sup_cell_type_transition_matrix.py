"""Calculates the transition rates of the cell types calculated from the first and last 10 time points."""

import matplotlib
import numpy
from matplotlib import pyplot as plt, colors
from matplotlib.axes import Axes
import matplotlib.image
from numpy import ndarray
from typing import List, Optional, NamedTuple, Tuple

import lib_data
import lib_figures
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io

_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
STEM_TO_ENTEROCYTE_BINS = 20

_CELL_TYPE_MEASUREMENT_TIME_POINTS = 10


class CellTypeInMatrix(NamedTuple):
    stem_to_ec_start: float  # Number between 0 and 1
    stem_to_ec_end: float  # Number between 0 and 1
    includes_paneth: bool

    def color(self) -> Tuple[float, float, float]:
        cell_type_names = ["STEM", "ENTEROCYTE", "PANETH"]
        if self.includes_paneth:
            cell_type_probabilities = [0, 0, 1]
        else:
            stem_cell_probability = (self.stem_to_ec_start + self.stem_to_ec_end) / 2
            cell_type_probabilities = [stem_cell_probability, 1 - stem_cell_probability, 0]

        return lib_figures.get_mixed_cell_type_color(cell_type_names, cell_type_probabilities)


class CellTypeTransitionMatrix:
    _count_matrix: ndarray  # First _STEM_TO_ENTEROCYTE_BINS rows and columns are for stem to enterocyte transitions

    # Last row and column are for Paneth cells
    # In between there's a column and row filled with NaNs, for visually separating both regions

    def __init__(self):
        self._count_matrix = numpy.zeros((STEM_TO_ENTEROCYTE_BINS + 3, STEM_TO_ENTEROCYTE_BINS + 3),
                                         dtype=numpy.float32)
        self._count_matrix[1, :] = numpy.nan
        self._count_matrix[:, 1] = numpy.nan

    def add_position(self, cell_types: List[str], probabilities_start: List[float], probabilities_end: List[float]):
        # Find cell types at start and end
        stem_to_ec_position_start = lib_data.find_stem_to_ec_location(cell_types, probabilities_start)
        is_paneth_start = cell_types[numpy.argmax(probabilities_start)] == "PANETH"
        stem_to_ec_position_end = lib_data.find_stem_to_ec_location(cell_types, probabilities_end)
        is_paneth_end = cell_types[numpy.argmax(probabilities_end)] == "PANETH"

        # Find bins based on cell types
        bin_start = 0 if is_paneth_start else None
        bin_end = 0 if is_paneth_end else None
        if stem_to_ec_position_start is not None:
            bin_start = 2 + int(stem_to_ec_position_start * STEM_TO_ENTEROCYTE_BINS)
        if stem_to_ec_position_end is not None:
            bin_end = 2 + int(stem_to_ec_position_end * STEM_TO_ENTEROCYTE_BINS)

        # If we have bins for the cell types, increment the count matrix
        if bin_start is not None and bin_end is not None:
            self._count_matrix[bin_start, bin_end] += 1

    @property
    def count_matrix(self) -> ndarray:
        return self._count_matrix

    @property
    def count_matrix_stem_to_ec(self) -> ndarray:
        return self._count_matrix[2:2 + STEM_TO_ENTEROCYTE_BINS, 2:2 + STEM_TO_ENTEROCYTE_BINS]

    def get_transition_count(self, from_cell_count: CellTypeInMatrix, to_cell_type: CellTypeInMatrix) -> int:
        """Gets the amount of transitions from one cell type to another."""
        bin_start_1 = 2 + int(from_cell_count.stem_to_ec_start * STEM_TO_ENTEROCYTE_BINS)
        bin_start_2 = 2 + int(from_cell_count.stem_to_ec_end * STEM_TO_ENTEROCYTE_BINS)
        bins_start = list(range(bin_start_1, bin_start_2))
        if from_cell_count.includes_paneth:
            bins_start.append(0)

        bin_end_1 = 2 + int(to_cell_type.stem_to_ec_start * STEM_TO_ENTEROCYTE_BINS)
        bin_end_2 = 2 + int(to_cell_type.stem_to_ec_end * STEM_TO_ENTEROCYTE_BINS)
        bins_end = list(range(bin_end_1, bin_end_2))
        if to_cell_type.includes_paneth:
            bins_end.append(0)

        counts = 0
        for bin_start in bins_start:
            for bin_end in bins_end:
                counts += self._count_matrix[bin_start, bin_end]
        return counts


def _find_first_track(track: LinkingTrack) -> LinkingTrack:
    previous_tracks = track.get_previous_tracks()
    while len(previous_tracks) > 0:
        track = previous_tracks.pop()
        previous_tracks = track.get_previous_tracks()
    return track


def main():
    transition_matrix = calculate_transition_matrix(_DATA_FILE_CONTROL)

    figure = lib_figures.new_figure(size=(5, 3.5))
    ax_matrix, ax_scalebar = figure.subplots(nrows=1, ncols=2, width_ratios=[1, 0.03])

    scaleable = plot_transition_matrix(ax_matrix, transition_matrix)
    figure.colorbar(scaleable, cax=ax_scalebar).set_label("Cell count")
    figure.tight_layout()
    plt.show()


def plot_transition_matrix(ax: Axes, transition_matrix: CellTypeTransitionMatrix, *, show_text: bool = True,
                           show_middle_lines: bool = True) -> matplotlib.image.AxesImage:
    x_ticks = [0] + list(numpy.arange(STEM_TO_ENTEROCYTE_BINS + 1) + 1.5)
    y_ticks = [0] + list(numpy.arange(STEM_TO_ENTEROCYTE_BINS + 1) + 1.5)
    x_ticklabels = ["Paneth"] + [(f"{i / STEM_TO_ENTEROCYTE_BINS:.1f}" if i % 2 == 0 else "") for i in
                                 range(STEM_TO_ENTEROCYTE_BINS + 1)]
    y_ticklabels = ["Paneth"] + [(f"{i / STEM_TO_ENTEROCYTE_BINS:.1f}" if i % 2 == 0 else "") for i in
                                 range(STEM_TO_ENTEROCYTE_BINS + 1)]

    ax.set_xlabel("End cell type (stem to EC)")
    ax.set_ylabel("Start cell type (stem to EC)")
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)
    ax.set_ylim(STEM_TO_ENTEROCYTE_BINS + 1.5, -0.5)
    ax.set_xlim(STEM_TO_ENTEROCYTE_BINS + 1.5, -0.5)
    if show_middle_lines:
        ax.axhline(y=STEM_TO_ENTEROCYTE_BINS * .5 + 1.5, color="white", linewidth=2)
        ax.axvline(x=STEM_TO_ENTEROCYTE_BINS * .5 + 1.5, color="white", linewidth=2)
    cmap = matplotlib.cm.gray
    cmap.set_bad("black")

    count_matrix = transition_matrix.count_matrix
    displayed_count_matrix = numpy.copy(count_matrix)
    vmax = int(numpy.nanmax(displayed_count_matrix)) + 1
    displayed_count_matrix[numpy.isnan(displayed_count_matrix)] = vmax  # Make those boxes white
    scaleable = ax.imshow(displayed_count_matrix, norm=colors.LogNorm(vmax=vmax),
                          cmap=cmap, interpolation="nearest")

    if show_text:
        for x in range(count_matrix.shape[1]):
            for y in range(count_matrix.shape[0]):
                if count_matrix[y, x] == 0 or numpy.isnan(count_matrix[y, x]):
                    continue
                ax.text(x, y, f"{count_matrix[y, x]:.0f}", ha="center", va="center",
                        color="white" if count_matrix[y, x] < vmax * 0.1 else "black")

    # Create color image for the cell types
    color_image = numpy.zeros((STEM_TO_ENTEROCYTE_BINS, 1, 3), dtype=numpy.float32)
    for i in range(STEM_TO_ENTEROCYTE_BINS):
        color_image[i, 0, :] = lib_figures.get_stem_to_ec_color(i / STEM_TO_ENTEROCYTE_BINS)
    ax.imshow(color_image, extent=(
        STEM_TO_ENTEROCYTE_BINS + 1.5, STEM_TO_ENTEROCYTE_BINS + 0.5, STEM_TO_ENTEROCYTE_BINS + 1.5, 1.5),
              aspect="auto")
    ax.imshow(color_image.reshape((1, STEM_TO_ENTEROCYTE_BINS, 3)), extent=(
        1.5, STEM_TO_ENTEROCYTE_BINS + 1.5, STEM_TO_ENTEROCYTE_BINS + 1.5, STEM_TO_ENTEROCYTE_BINS + 0.5),
              aspect="auto")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return scaleable


def calculate_transition_matrix(data_file: str) -> CellTypeTransitionMatrix:
    transition_matrix = CellTypeTransitionMatrix()
    for experiment in list_io.load_experiment_list_file(data_file):
        position_data = experiment.position_data
        cell_types = experiment.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise ValueError(f"Cell type probabilities not found in experiment {experiment.name}")

        for position_end in experiment.positions.of_time_point(experiment.positions.last_time_point()):
            track_end = experiment.links.get_track(position_end)
            if track_end is None:
                continue
            track_start = _find_first_track(track_end)
            if track_start.first_time_point() != experiment.positions.first_time_point():
                continue

            # Found a cell that spans the entire experiment
            cell_probabilities_start = _find_cell_probabilities(position_data, track_start, at_start=True)
            cell_probabilities_end = _find_cell_probabilities(position_data, track_end, at_end=True)
            if cell_probabilities_start is None or cell_probabilities_end is None:
                continue

            transition_matrix.add_position(cell_types, cell_probabilities_start, cell_probabilities_end)
    return transition_matrix


def _find_cell_probabilities(position_data: PositionData, track: LinkingTrack, *, at_start: bool = False,
                             at_end: bool = False) -> Optional[List[float]]:
    if (at_start and at_end) or (not at_start and not at_end):
        raise ValueError("Exactly one of at_start and at_end must be True")

    cell_type_probabilities_sum: Optional[ndarray] = None
    cell_type_probabilities_count = 0

    for position in track.positions():
        # Time range check
        if at_start:
            if position.time_point_number() > track.first_time_point_number() + _CELL_TYPE_MEASUREMENT_TIME_POINTS:
                break
        else:
            if position.time_point_number() < track.last_time_point_number() - _CELL_TYPE_MEASUREMENT_TIME_POINTS:
                continue

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
    return cell_type_probabilities_sum / cell_type_probabilities_count


if __name__ == "__main__":
    main()
