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

import numpy
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
_CRYPT_BOTTOM_CUTOFF_UM = 15

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
            return 0
        return self.has_edu_and_brdu_stain_count / self.has_edu_stain_count * 100

    def __add__(self, other: Any) -> Union[NotImplementedType, "_CellCount"]:
        if not isinstance(other, _CellCount):
            return NotImplemented
        summed_cell_count = _CellCount()
        summed_cell_count.has_edu_stain_count = self.has_edu_stain_count + other.has_edu_stain_count
        summed_cell_count.has_edu_and_brdu_stain_count = self.has_edu_and_brdu_stain_count + other.has_edu_and_brdu_stain_count
        return summed_cell_count


class _CountsByTimePoints:
    _counts_crypt_top: List[_CellCount]  # Indexed like _MEASUREMENT_TIME_POINTS_H
    _counts_crypt_bottom: List[_CellCount]  # Also indexed like _MEASUREMENT_TIME_POINTS_H
    name: str

    def __init__(self, name: str):
        self.name = name
        self._counts_crypt_top =[_CellCount() for _ in range(len(_MEASUREMENT_TIME_POINTS_H))]
        self._counts_crypt_bottom = [_CellCount() for _ in range(len(_MEASUREMENT_TIME_POINTS_H))]

    def add_edu_stained_cell(self, time_h: int, height_in_crypt_um: float, *, has_edu_and_brdu_stain: bool):
        time_index = _MEASUREMENT_TIME_POINTS_H.index(time_h)
        if height_in_crypt_um < _CRYPT_BOTTOM_CUTOFF_UM:
            self._counts_crypt_bottom[time_index].has_edu_stain_count += 1
            if has_edu_and_brdu_stain:
                self._counts_crypt_bottom[time_index].has_edu_and_brdu_stain_count += 1
        else:
            self._counts_crypt_top[time_index].has_edu_stain_count += 1
            if has_edu_and_brdu_stain:
                self._counts_crypt_top[time_index].has_edu_and_brdu_stain_count += 1

    def percentages(self, *, crypt_bottom: bool) -> List[float]:
        if crypt_bottom:
            return [count.percentage() for count in self._counts_crypt_bottom]
        else:
            return [count.percentage() for count in self._counts_crypt_top]

    def edu_cell_counts(self, *, crypt_bottom: bool) -> List[int]:
        if crypt_bottom:
            return [count.has_edu_stain_count for count in self._counts_crypt_bottom]
        else:
            return [count.has_edu_stain_count for count in self._counts_crypt_top]

    def __add__(self, other: Any) -> Union[NotImplementedType, "_CountsByTimePoints"]:
        if not isinstance(other, _CountsByTimePoints):
            return NotImplemented
        new_counts = _CountsByTimePoints("sum")
        for i in range(len(self._counts_crypt_top)):
            new_counts._counts_crypt_top[i] = self._counts_crypt_top[i] + other._counts_crypt_top[i]
            new_counts._counts_crypt_bottom[i] = self._counts_crypt_bottom[i] + other._counts_crypt_bottom[i]
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

    figure = lib_figures.new_figure()
    ax_ta, ax_stem = figure.subplots(nrows=2, ncols=1, sharex=True)
    _draw_bars(ax_ta, all_counts_by_time_point, crypt_bottom=False)
    _draw_bars(ax_stem, all_counts_by_time_point, crypt_bottom=True)

    ax_stem.set_xticks(range(len(_MEASUREMENT_TIME_POINTS_H)), [f"{time_point}h" for time_point in _MEASUREMENT_TIME_POINTS_H])
    ax_ta.set_ylabel("Edu+BrdU+/Edu+ (%)")
    ax_ta.set_title("Other cells")
    ax_stem.set_title(f"Crypt bottom (<{_CRYPT_BOTTOM_CUTOFF_UM} µm)")
    plt.show()


def _draw_bars(ax: Axes, all_counts_by_time_point: List[_CountsByTimePoints], *, crypt_bottom: bool):
    random = numpy.random.Generator(numpy.random.MT19937(seed=1))
    for counts_by_time_point in all_counts_by_time_point:
        percentages = counts_by_time_point.percentages(crypt_bottom=crypt_bottom)
        x_positions = numpy.arange(len(_MEASUREMENT_TIME_POINTS_H)) + random.normal(0, 0.15, len(_MEASUREMENT_TIME_POINTS_H))
        ax.scatter(x_positions, percentages,
                   color="black", zorder=5, alpha=0.6, linewidths=0, s=10, marker="s")
        # ax.text(len(_MEASUREMENT_TIME_POINTS_H) - 1, percentages[-1], counts_by_time_point.name, ha="left", va="center")
    summed_counts_by_time_point = sum(all_counts_by_time_point, _CountsByTimePoints("sum"))

    # Draw bars and counts
    percentages = summed_counts_by_time_point.percentages(crypt_bottom=crypt_bottom)
    edu_counts = summed_counts_by_time_point.edu_cell_counts(crypt_bottom=crypt_bottom)
    ax.bar(range(len(_MEASUREMENT_TIME_POINTS_H)), percentages, color="#74b9ff")
    for i in range(len(_MEASUREMENT_TIME_POINTS_H)):
        ax.text(i, percentages[i] + 1, f"N$_{{EdU}}$={edu_counts[i]}", ha="center", va="top")


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
            resolution = experiment.images.resolution()

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
                    spline_position_um = spline_position.pos * resolution.pixel_size_x_um

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
                    counts_by_time_point.add_edu_stained_cell(time_h, spline_position_um, has_edu_and_brdu_stain=has_brdu_staining)

            all_counts_by_time_point.append(counts_by_time_point)
    return all_counts_by_time_point


if __name__ == "__main__":
    main()
