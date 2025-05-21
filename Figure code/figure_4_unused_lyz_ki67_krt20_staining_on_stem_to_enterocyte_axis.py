from collections import defaultdict
from enum import Enum, auto
from typing import List, Union, Tuple

import matplotlib
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray

import lib_data
import lib_figures
from organoid_tracker.core import MPLColor, Name

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers, intensity_calculator

_DATA_FILE_STAINING = "../../Data/Live and immunostaining multiple time points/Immunostaining all.autlist"
_DATA_FILE_LIVE = "../../Data/Live and immunostaining multiple time points/Live all.autlist"

_MIN_COUNT_IN_BIN = 5


class _ExperimentCondition(Enum):
    YOUNG_CONTROL = auto()
    OLD_CONTROL = auto()
    DIRECTLY_AFTER_DT = auto()
    AFTER_DT_8H = auto()
    AFTER_DT_24H = auto()
    AFTER_DT_48H = auto()

    @staticmethod
    def from_letter(letter: str) -> "_ExperimentCondition":
        """Gets the condition from the single uppercase letter."""
        if letter == "A":
            return _ExperimentCondition.YOUNG_CONTROL
        elif letter == "B":
            return _ExperimentCondition.OLD_CONTROL
        elif letter == "C":
            return _ExperimentCondition.DIRECTLY_AFTER_DT
        elif letter == "D":
            return _ExperimentCondition.AFTER_DT_8H
        elif letter == "E":
            return _ExperimentCondition.AFTER_DT_24H
        elif letter == "F":
            return _ExperimentCondition.AFTER_DT_48H
        else:
            raise ValueError(f"Unknown experiment condition {letter}")

    def display_name(self) -> str:
        return self.name.replace("_", " ").lower()

def _get_counts_positive_and_total(bins: ndarray, locations: Union[ndarray, List[float]],
                                   boolean_array: Union[ndarray, List[bool]]) -> Tuple[ndarray, ndarray]:
    total_count_by_bin = numpy.zeros(len(bins) - 1, dtype=int)
    positive_count_by_bin = numpy.zeros(len(bins) - 1, dtype=int)
    for is_positive, location in zip(boolean_array, locations):
        bin_index = numpy.searchsorted(bins, location, side='left') - 1
        if 0 <= bin_index < len(total_count_by_bin):
            total_count_by_bin[bin_index] += 1
            if is_positive:
                positive_count_by_bin[bin_index] += 1
    return positive_count_by_bin, total_count_by_bin


class _Lgr5Data:
    _is_lgr5_positive: List[bool]
    _stem_to_enterocyte_position: List[float]

    def __init__(self):
        self._is_lgr5_positive = []
        self._stem_to_enterocyte_position = []

    def add_cell(self, is_lgr5_positive: bool, stem_to_enterocyte_position: float):
        self._is_lgr5_positive.append(is_lgr5_positive)
        self._stem_to_enterocyte_position.append(stem_to_enterocyte_position)

    def get_fraction_positive(self, bins: ndarray) -> ndarray:
        positive, total = _get_counts_positive_and_total(bins, locations=self._stem_to_enterocyte_position,
                                                         boolean_array=self._is_lgr5_positive)
        total[total == 0] = 1  # Avoid division by zero
        return positive / total

    def get_count_positive_and_negative(self, bins: ndarray) -> Tuple[ndarray, ndarray]:
        positives, totals = _get_counts_positive_and_total(bins, locations=self._stem_to_enterocyte_position,
                                                           boolean_array=self._is_lgr5_positive)
        return positives, totals - positives


class _StainedData:
    _is_ki67_positive: List[bool]
    _is_krt20_positive: List[bool]
    _stem_to_enterocyte_position: List[float]

    def __init__(self):
        self._is_ki67_positive = []
        self._is_krt20_positive = []
        self._stem_to_enterocyte_position = []

    def add_cell(self, *, is_ki67_positive: bool, is_krt20_positive: bool, stem_to_enterocyte_position: float):
        self._is_ki67_positive.append(is_ki67_positive)
        self._is_krt20_positive.append(is_krt20_positive)
        self._stem_to_enterocyte_position.append(stem_to_enterocyte_position)

    def get_ki67_count_positive_and_negative(self, bins: ndarray) -> Tuple[ndarray, ndarray]:
        positives, totals = _get_counts_positive_and_total(bins, locations=self._stem_to_enterocyte_position,
                                                           boolean_array=self._is_ki67_positive)
        return positives, totals - positives

    def get_krt20_count_positive_and_negative(self, bins: ndarray) -> Tuple[ndarray, ndarray]:
        positives, totals = _get_counts_positive_and_total(bins, locations=self._stem_to_enterocyte_position,
                                                           boolean_array=self._is_krt20_positive)
        return positives, totals - positives


def _analyze_fixed_types(experiment_stained: Experiment, *, into: _StainedData) -> bool:
    cell_types = experiment_stained.global_data.get_data("ct_probabilities")
    if cell_types is None:
        return False

    if not experiment_stained.position_data.has_position_data_with_name("type"):
        return False  # No cell type annotations

    for position in experiment_stained.positions:
        # Follows the cell type annotation as described in the README.txt in the data folder
        position_type = position_markers.get_position_type(experiment_stained.position_data, position)
        ki67_positive = position_type == "STEM_PUTATIVE" or position_type == "ABSORPTIVE_PRECURSOR"
        krt20_positive = position_type == "ENTEROCYTE" or position_type == "ABSORPTIVE_PRECURSOR"

        ct_probabilities = experiment_stained.position_data.get_position_data(position, "ct_probabilities")
        if ct_probabilities is None:
            continue

        stem_to_enterocyte_position = lib_data.find_stem_to_ec_location(cell_types, ct_probabilities)
        if stem_to_enterocyte_position is None:
            continue  # Not on the stem to enterocyte axis

        into.add_cell(is_ki67_positive=ki67_positive, is_krt20_positive=krt20_positive,
                      stem_to_enterocyte_position=stem_to_enterocyte_position)
    return True


def _analyze_lgr5(experiment_live: Experiment, *, into: _Lgr5Data):
    cell_types = experiment_live.global_data.get_data("ct_probabilities")
    for position in experiment_live.positions:

        # Follows the cell type annotation as described in the README.txt in the data folder
        position_type = position_markers.get_position_type(experiment_live.position_data, position)
        lgr5_positive = position_type == "STEM"

        ct_probabilities = experiment_live.position_data.get_position_data(position, "ct_probabilities")
        if ct_probabilities is None:
            continue

        stem_to_enterocyte_position = lib_data.find_stem_to_ec_location(cell_types, ct_probabilities)
        if stem_to_enterocyte_position is None:
            continue  # Not on the stem to enterocyte axis

        into.add_cell(lgr5_positive, stem_to_enterocyte_position)


def _get_ratio_positive(counts_positive: ndarray, counts_negative: ndarray) -> ndarray:
    totals = counts_positive + counts_negative

    # Avoid division by zero
    totals_no_zero = totals.copy()
    totals_no_zero[totals_no_zero == 0] = 1
    fractions = counts_positive / totals_no_zero
    fractions[totals < _MIN_COUNT_IN_BIN] = numpy.nan  # Set fractions to NaN where totals are too low
    return fractions


def _parse_experiment_name(name: Name) -> str:
    """Parses the experiment name into a standardized format, to allow matching experiments."""
    name_str = name.get_name()
    name_str = name_str.replace("Position", "Pos")
    name_str_parts = name_str.split(" » ", maxsplit=1)
    if len(name_str_parts) != 2:
        return name_str  # Cannot parse, return original name
    return name_str_parts[0][0:2] + " » " + name_str_parts[1]


def main():
    experiments_live = list(list_io.load_experiment_list_file(_DATA_FILE_LIVE, load_images=False))
    experiments_stained = list(list_io.load_experiment_list_file(_DATA_FILE_STAINING, load_images=False))

    lgr5_data_all = defaultdict(_Lgr5Data)
    stained_data_all = defaultdict(_StainedData)

    for experiment_stained in experiments_stained:
        experiment_stained_name = _parse_experiment_name(experiment_stained.name)
        for experiment_live in experiments_live:
            experiment_live_name = _parse_experiment_name(experiment_live.name)
            if experiment_stained_name != experiment_live_name:
                continue

            condition = _ExperimentCondition.from_letter(experiment_live_name[0])
            if _analyze_fixed_types(experiment_stained, into=stained_data_all[condition]):
                _analyze_lgr5(experiment_live, into=lgr5_data_all[condition])
                print("Matching data: ", condition, experiment_live.name, "---with---", experiment_stained.name)
                break

    figure = lib_figures.new_figure(size=(18, 10))
    axes = figure.subplots(len(_ExperimentCondition), 3, sharex=True)

    axes[0, 0].set_title("KRT20 positive fraction")
    axes[0, 1].set_title("Ki67 positive fraction")
    axes[0, 2].set_title("Lgr5 positive fraction")

    for condition, axes_row in zip(_ExperimentCondition, axes):
        stained_data_cond = stained_data_all[condition]
        lgr5_data_cond = lgr5_data_all[condition]
        ax_krt20_fraction, ax_ki67_fraction, ax_lgr5_fraction = axes_row

        ax_krt20_fraction.text(0.05, 0.9, condition.display_name(), ha='left', va='top', fontsize=12, transform=ax_krt20_fraction.transAxes)

        bins = numpy.arange(0, 1.05, 0.05)
        krt20_positive, krt20_negative = stained_data_cond.get_krt20_count_positive_and_negative(bins)
        krt20_positive_fraction = _get_ratio_positive(krt20_positive, krt20_negative)
        bin_colors = [lib_figures.get_stem_to_ec_color((bins[i] + bins[i + 1]) / 2) for i in range(len(bins) - 1)]
        ax_krt20_fraction.set_ylabel("KRT20 positive cells (%)")
        ax_krt20_fraction.bar(bins[:-1], krt20_positive_fraction * 100, width=0.05, align='edge', color=bin_colors)
        trans = matplotlib.transforms.blended_transform_factory(ax_krt20_fraction.transData, ax_krt20_fraction.transAxes)
        for i in range(len(bins) - 1):
            if krt20_positive[i] + krt20_negative[i] >= _MIN_COUNT_IN_BIN:
                ax_krt20_fraction.text((bins[i] + bins[i + 1]) / 2, 1, str(krt20_positive[i] + krt20_negative[i]),
                                       ha='center', va='top', fontsize=8, transform=trans)
        ax_krt20_fraction.set_ylim(0, 110)

        ki67_positive, ki67_negative = stained_data_cond.get_ki67_count_positive_and_negative(bins)
        ki67_positive_fraction = _get_ratio_positive(ki67_positive, ki67_negative)
        ax_ki67_fraction.bar(bins[:-1], ki67_positive_fraction * 100, width=0.05, align='edge', color=bin_colors)
        ax_ki67_fraction.set_ylabel("Positive cells (%)")
        trans = matplotlib.transforms.blended_transform_factory(ax_ki67_fraction.transData, ax_ki67_fraction.transAxes)
        for i in range(len(bins) - 1):
            if ki67_positive[i] + ki67_negative[i] >= _MIN_COUNT_IN_BIN:
                ax_ki67_fraction.text((bins[i] + bins[i + 1]) / 2, 1, str(ki67_positive[i] + ki67_negative[i]),
                             ha='center', va='top', fontsize=8, transform=trans)
        ax_ki67_fraction.set_ylim(0, 110)

        lgr5_positive, lgr5_negative = lgr5_data_cond.get_count_positive_and_negative(bins)
        lgr5_positive_fraction = _get_ratio_positive(lgr5_positive, lgr5_negative)
        ax_lgr5_fraction.bar(bins[:-1], lgr5_positive_fraction * 100, width=0.05, align='edge', color=bin_colors)

        # Draw text near the top of the axes
        trans = matplotlib.transforms.blended_transform_factory(
            ax_lgr5_fraction.transData, ax_lgr5_fraction.transAxes)
        for i in range(len(bins) - 1):
            if lgr5_positive[i] + lgr5_negative[i] >= _MIN_COUNT_IN_BIN:
                ax_lgr5_fraction.text((bins[i] + bins[i + 1]) / 2, 1, str(lgr5_positive[i] + lgr5_negative[i]),
                             ha='center', va='top', fontsize=8, transform=trans)
        ax_lgr5_fraction.set_xlabel("Stem-to-enterocyte axis")
        ax_lgr5_fraction.set_ylabel("Positive cells (%)")

        ax_lgr5_fraction.set_xlim(0.78, 0.1)
        ax_lgr5_fraction.set_ylim(0, 110)

        # Hide top and right spines
        for ax in [ax_krt20_fraction, ax_ki67_fraction, ax_lgr5_fraction]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
