# In https://doi.org/10.1016/j.devcel.2024.12.023 - Figure 2E-H, EdU and BrdU cell cycle measurement
# Their experimental timeline:
# EdU pulse at t = 0h
# First measurement at t = 1h
# Continous BrdU from t = 9h
# Additional measurements at t = 10h, t = 15h, t = 24h
#
# According to ChatGPT: ("How long does the S phase of the cell cycle last in an intestinal epithelial cell?")
# G1 phase: ~2–6 hours
# S phase: ~6–8 hours
# G2 phase: ~2–4 hours
# M phase: ~1 hour
# So S phase is about four hours before mitosis.
# We want to simulate this experiment with our tracking data. We cannot see which cells are in the S phase. As an
# estimation, we could check which cells would divide within 11 to 4 hours. Or equivalently (if we make the same shift
# in the next generation), which cells would divide within 7 to 0 hours.
from types import NotImplementedType
from typing import Iterable, NamedTuple, List, Union, Any, Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io

import lib_figures
import lib_data

_DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"
_MEASUREMENT_TIME_POINTS_H = [1, 10, 15, 24]
_BRDU_ADDITION_TIME_H = 9
_STEM_CELL_THRESHOLD = 0.55  # Everything above this is considered a stem cell

_TIME_NOT_NEXT_MITOSIS = 7


class _CellCount:
    has_edu_stain_count: int
    has_edu_and_brdu_stain_count: int

    def __init__(self):
        self.has_edu_stain_count = 0
        self.has_edu_and_brdu_stain_count = 0

    def percentage(self) -> float:
        """Gets the percentage of single-stained cells that are double-stained."""
        if self.has_edu_stain_count == 0:
            return float("NaN")
        return self.has_edu_and_brdu_stain_count / self.has_edu_stain_count * 100

    def __add__(self, other: Any) -> Union[NotImplementedType, "_CellCount"]:
        if not isinstance(other, _CellCount):
            return NotImplemented
        summed_cell_count = _CellCount()
        summed_cell_count.has_edu_stain_count = self.has_edu_stain_count + other.has_edu_stain_count
        summed_cell_count.has_edu_and_brdu_stain_count = self.has_edu_and_brdu_stain_count + other.has_edu_and_brdu_stain_count
        return summed_cell_count


class _CountsByTimePoints:
    _name: str
    _counts_stem_cells: List[_CellCount]  # Indexed like _MEASUREMENT_TIME_POINTS_H
    _counts_ta_cells: List[_CellCount]  # Also indexed like _MEASUREMENT_TIME_POINTS_H

    def __init__(self, name: str):
        self._name = name
        self._counts_stem_cells =[_CellCount() for _ in range(len(_MEASUREMENT_TIME_POINTS_H))]
        self._counts_ta_cells = [_CellCount() for _ in range(len(_MEASUREMENT_TIME_POINTS_H))]

    def add_edu_stained_cell(self, time_h: int, stem_to_ec_index: float, *, has_edu_and_brdu_stain: bool):
        time_index = _MEASUREMENT_TIME_POINTS_H.index(time_h)
        if stem_to_ec_index < _STEM_CELL_THRESHOLD:
            self._counts_ta_cells[time_index].has_edu_stain_count += 1
            if has_edu_and_brdu_stain:
                self._counts_ta_cells[time_index].has_edu_and_brdu_stain_count += 1
        else:
            self._counts_stem_cells[time_index].has_edu_stain_count += 1
            if has_edu_and_brdu_stain:
                self._counts_stem_cells[time_index].has_edu_and_brdu_stain_count += 1

    def percentages(self, *, stem: bool) -> List[float]:
        if stem:
            return [count.percentage() for count in self._counts_stem_cells]
        else:
            return [count.percentage() for count in self._counts_ta_cells]

    def edu_cell_counts(self, *, stem: bool) -> List[int]:
        if stem:
            return [count.has_edu_stain_count for count in self._counts_stem_cells]
        else:
            return [count.has_edu_stain_count for count in self._counts_ta_cells]

    def __add__(self, other: Any) -> Union[NotImplementedType, "_CountsByTimePoints"]:
        if not isinstance(other, _CountsByTimePoints):
            return NotImplemented
        new_counts = _CountsByTimePoints("sum")
        for i in range(len(self._counts_stem_cells)):
            new_counts._counts_stem_cells[i] = self._counts_stem_cells[i] + other._counts_stem_cells[i]
            new_counts._counts_ta_cells[i] = self._counts_ta_cells[i] + other._counts_ta_cells[i]
        return new_counts


def _find_cells_soon_in_mitosis_at_start(experiment: Experiment) -> Iterable[Position]:
    """Finds positions in the first time point that will soon undergo mitosis."""
    for position in experiment.positions.of_time_point(experiment.positions.first_time_point()):
        if _will_go_into_mitosis(experiment, position):
            yield position


def _will_go_into_mitosis(experiment: Experiment, position: Position) -> bool:
    track = experiment.links.get_track(position)
    if track is None:
        return False
    if not track.will_divide():
        return False

    timings = experiment.images.timings()
    mitosis_time = timings.get_time_h_since_start(track.last_time_point_number())
    current_time = timings.get_time_h_since_start(position.time_point_number())
    if mitosis_time - current_time < _TIME_NOT_NEXT_MITOSIS:
        return True
    return False


def _find_time_point_since_start(experiment: Experiment, time_h: float) -> TimePoint:
    timings = experiment.images.timings()
    time_start_h = timings.get_time_h_since_start(experiment.positions.first_time_point_number())
    for time_point in experiment.positions.time_points():
        if timings.get_time_h_since_start(time_point) - time_start_h >= time_h:
            return time_point
    raise ValueError(f"No time point found at {time_h} hours since start - beyond the last time point.")


def main():
    experiments = list_io.load_experiment_list_file(_DATA_FILE, load_images=False)
    all_counts_by_time_point = _get_counts_by_time_point(experiments)
    for all_counts_by_time_point_single_crypt in all_counts_by_time_point:
        all_counts_by_time_point_single_crypt = [all_counts_by_time_point_single_crypt]

        figure = lib_figures.new_figure()
        figure.suptitle(all_counts_by_time_point_single_crypt[0]._name)
        ax_ta, ax_stem = figure.subplots(nrows=2, ncols=1, sharex=True)
        _draw_bars(ax_ta, all_counts_by_time_point_single_crypt, stem=False)
        _draw_bars(ax_stem, all_counts_by_time_point_single_crypt, stem=True)

        ax_stem.set_xticks(range(len(_MEASUREMENT_TIME_POINTS_H)), [f"{time_point}h" for time_point in _MEASUREMENT_TIME_POINTS_H])
        ax_ta.set_ylabel("Edu+BrdU+/Edu+ (%)")
        ax_ta.set_title("TA-like cells")
        ax_stem.set_title("Stem cells")
        plt.show()


def _draw_bars(ax: Axes, all_counts_by_time_point: List[_CountsByTimePoints], *, stem: bool):
    for counts_by_time_point in all_counts_by_time_point:
        ax.scatter(range(len(_MEASUREMENT_TIME_POINTS_H)), counts_by_time_point.percentages(stem=stem),
                      color="black", zorder=5, alpha=0.6, linewidths=0, s=10, marker="s")
    summed_counts_by_time_point = sum(all_counts_by_time_point, _CountsByTimePoints("sum"))

    # Draw bars and counts
    percentages = summed_counts_by_time_point.percentages(stem=stem)
    edu_counts = summed_counts_by_time_point.edu_cell_counts(stem=stem)
    ax.bar(range(len(_MEASUREMENT_TIME_POINTS_H)), percentages, color="#74b9ff")
    for i in range(len(_MEASUREMENT_TIME_POINTS_H)):
        ax.text(i, percentages[i] + 1, f"N$_{{EdU}}$={edu_counts[i]}", ha="center", va="bottom")


def _find_previous_mitosis_time(experiment: Experiment, position: Position) -> Optional[float]:
    track = experiment.links.get_track(position)
    if track is None:
        return None
    for parent_track in track.find_all_previous_tracks(include_self=False):
        if parent_track.will_divide():
            timings = experiment.images.timings()
            return timings.get_time_h_since_start(parent_track.last_time_point_number())
    return None


def _get_counts_by_time_point(experiments: Iterable[Experiment]) -> List[_CountsByTimePoints]:
    all_counts_by_time_point = list()
    for experiment in experiments:
        # Record all cells that got EdU staining
        cells_in_mitosis = set(_find_cells_soon_in_mitosis_at_start(experiment))

        for spline_index, _ in experiment.splines.of_time_point(experiment.positions.last_time_point()):
            counts_by_time_point = _CountsByTimePoints(f"{experiment.name}-{spline_index}")
            cell_types = experiment.global_data.get_data("ct_probabilities")
            if cell_types is None:
                raise ValueError("No cell type probabilities found in experiment data")

            for time_h in _MEASUREMENT_TIME_POINTS_H:

                time_point = _find_time_point_since_start(experiment, time_h)
                for position in experiment.positions.of_time_point(time_point):
                    position_original = experiment.links.get_position_at_time_point(position,
                                                                                    experiment.positions.first_time_point())
                    if position_original is None or position_original not in cells_in_mitosis:
                        continue  # Not interested in this cell - not originating from a EdU-stained cell

                    spline_position = experiment.splines.to_position_on_spline(position, only_axis=True)
                    if spline_position is None or spline_position.spline_id != spline_index:
                        continue  # Not interested in this cell - not on this crypt-villus axis

                    probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
                    stem_to_ec_location = lib_data.find_stem_to_ec_location(cell_types, probabilities)
                    if stem_to_ec_location is None:
                        continue  # Not interested in this cell - not on the stem-to-enterocyte axis

                    # Check for BrdU staining
                    has_brdu_staining = False
                    if time_h > _BRDU_ADDITION_TIME_H and _will_go_into_mitosis(experiment, position):
                        # If it divides soon, it will have BrdU staining (simulated moved S phase)
                        has_brdu_staining = True
                    else:
                        # If it has divided before, it will also have BrdU staining (as BrdU is added continuously)
                        previous_mitosis_time = _find_previous_mitosis_time(experiment, position)
                        if previous_mitosis_time is not None and previous_mitosis_time > _BRDU_ADDITION_TIME_H:
                            has_brdu_staining = True

                    # Add cell
                    counts_by_time_point.add_edu_stained_cell(time_h, stem_to_ec_location, has_edu_and_brdu_stain=has_brdu_staining)

            all_counts_by_time_point.append(counts_by_time_point)
    return all_counts_by_time_point


if __name__ == "__main__":
    main()
