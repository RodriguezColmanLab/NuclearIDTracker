from typing import List, Union, Tuple

import matplotlib
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray

import lib_data
import lib_figures
from organoid_tracker.core import MPLColor

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers, intensity_calculator

_DATA_FILE_STAINING = "../../Data/Live and immunostaining multiple time points/Immunostaining young versus old.autlist"
_DATA_FILE_LIVE = "../../Data/Live and immunostaining multiple time points/Live young versus old.autlist"

_MIN_COUNT_IN_BIN = 5


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


def _get_values_by_bin(bins: ndarray, locations: Union[ndarray, List[float]], value_array: Union[ndarray, List[float]]) -> List[List[float]]:
    values_by_bin = [[] for _ in range(len(bins) - 1)]
    for value, location in zip(value_array, locations):
        bin_index = numpy.searchsorted(bins, location, side='left') - 1
        if 0 <= bin_index < len(values_by_bin):
            values_by_bin[bin_index].append(value)

    return values_by_bin


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


class _Ki67Data:
    _is_ki67_positive: List[bool]
    _ki67_intensity: List[float]
    _stem_to_enterocyte_position: List[float]

    def __init__(self):
        self._is_ki67_positive = []
        self._ki67_intensity = []
        self._stem_to_enterocyte_position = []

    def add_cell(self, is_ki67_positive: bool, ki67_intensity: float, stem_to_enterocyte_position: float):
        self._is_ki67_positive.append(is_ki67_positive)
        self._ki67_intensity.append(ki67_intensity)
        self._stem_to_enterocyte_position.append(stem_to_enterocyte_position)

    def get_count_positive_and_negative(self, bins: ndarray) -> Tuple[ndarray, ndarray]:
        positives, totals = _get_counts_positive_and_total(bins, locations=self._stem_to_enterocyte_position,
                                                           boolean_array=self._is_ki67_positive)
        return positives, totals - positives

    def get_ki67_values(self, bins: ndarray) -> List[List[float]]:
        return _get_values_by_bin(bins, locations=self._stem_to_enterocyte_position,
                                        value_array=self._ki67_intensity)



def _analyze_ki67(experiment_stained: Experiment, *, into: _Ki67Data) -> bool:
    cell_types = experiment_stained.global_data.get_data("ct_probabilities")
    if cell_types is None:
        return False

    added_data = False
    for position in experiment_stained.positions:

        # Follows the cell type annotation as described in the README.txt in the data folder
        position_type = position_markers.get_position_type(experiment_stained.position_data, position)
        ki67_positive = position_type == "STEM_PUTATIVE" or position_type == "ABSORPTIVE_PRECURSOR"

        ki67_intensity = intensity_calculator.get_normalized_intensity(experiment_stained, position, intensity_key="intensity_ki67", per_pixel=True)
        ct_probabilities = experiment_stained.position_data.get_position_data(position, "ct_probabilities")
        if ct_probabilities is None:
            continue

        stem_to_enterocyte_position = lib_data.find_stem_to_ec_location(cell_types, ct_probabilities)
        if stem_to_enterocyte_position is None:
            continue  # Not on the stem to enterocyte axis

        into.add_cell(ki67_positive, ki67_intensity, stem_to_enterocyte_position)
        added_data = True
    return added_data


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


def _plot_violins(ax: Axes, bins: ndarray, ki67_values: List[List[float]], bin_colors: List[MPLColor]):
    x_positions = list()
    ki67_values_filtered = list()
    bin_colors_filtered = list()
    for i in range(len(bins) - 1):
        if len(ki67_values[i]) >= _MIN_COUNT_IN_BIN:
            x_positions.append((bins[i] + bins[i + 1]) / 2)
            ki67_values_filtered.append(ki67_values[i])
            bin_colors_filtered.append(bin_colors[i])

    violin = ax.violinplot(ki67_values_filtered, positions=x_positions, showmeans=False, showmedians=True,
                  showextrema=False, widths=bins[1] - bins[0],)
    for color, body in zip(bin_colors_filtered, violin['bodies']):
        body.set_facecolor(color)
        body.set_linewidth(0)
        body.set_alpha(1)
    # generator = numpy.random.Generator(numpy.random.MT19937(seed=1))
    # for i in range(len(bins) - 1):
    #     if len(ki67_values[i]) >= _MIN_COUNT_IN_BIN:
    #         x_position = (bins[i] + bins[i + 1]) / 2
    #         ax.scatter(generator.normal(x_position, 0.01, size=len(ki67_values[i])), ki67_values[i], color=bin_colors[i], linewidth=0, s=2)


def main():
    experiments_live = list(list_io.load_experiment_list_file(_DATA_FILE_LIVE, load_images=False))
    experiments_stained = list(list_io.load_experiment_list_file(_DATA_FILE_STAINING, load_images=False))

    lgr5_data = _Lgr5Data()
    ki67_data = _Ki67Data()

    for experiment_stained in experiments_stained:
        for experiment_live in experiments_live:
            if experiment_stained.name.get_name() != experiment_live.name.get_name():
                continue
            if not experiment_stained.name.get_name().startswith("B"):
                # Needs to be of row B, where the old organoids are
                continue

            print("Found matching experiment: ", experiment_stained.name.get_name())
            if _analyze_ki67(experiment_stained, into=ki67_data):
                _analyze_lgr5(experiment_live, into=lgr5_data)

    figure = lib_figures.new_figure(size=(3.5, 4.5))
    ax_ki67_mean, ax_ki67_fraction, ax_lgr5 = figure.subplots(3, 1, sharex=True)
    ax_lgr5: Axes

    bins = numpy.arange(0, 1.05, 0.05)
    ax_ki67_mean.set_title("Ki67 mean intensity")
    ki67_positive, ki67_negative = ki67_data.get_count_positive_and_negative(bins)
    ki67_values = ki67_data.get_ki67_values(bins)
    bin_colors = [lib_figures.get_stem_to_ec_color((bins[i] + bins[i + 1]) / 2) for i in range(len(bins) - 1)]
    _plot_violins(ax_ki67_mean, bins, ki67_values, bin_colors)
    ax_ki67_mean.set_ylabel("Intensity per pixel")
    trans = matplotlib.transforms.blended_transform_factory(
        ax_ki67_mean.transData, ax_ki67_mean.transAxes)
    for i in range(len(bins) - 1):
        if ki67_positive[i] + ki67_negative[i] >= _MIN_COUNT_IN_BIN:
            ax_ki67_mean.text((bins[i] + bins[i + 1]) / 2, 1, str(ki67_positive[i] + ki67_negative[i]),
                         ha='center', va='top', fontsize=8, transform=trans)

    ax_ki67_fraction.set_title("Ki67 positive fraction")
    ki67_positive_fraction = _get_ratio_positive(ki67_positive, ki67_negative)
    ax_ki67_fraction.bar(bins[:-1], ki67_positive_fraction * 100, width=0.05, align='edge', color=bin_colors)
    ax_ki67_fraction.set_ylabel("Positive cells (%)")
    ax_ki67_fraction.set_ylim(0, 85)

    ax_lgr5.set_title("Lgr5 positive fraction")
    lgr5_positive, lgr5_negative = lgr5_data.get_count_positive_and_negative(bins)
    lgr5_positive_fraction = _get_ratio_positive(lgr5_positive, lgr5_negative)
    ax_lgr5.bar(bins[:-1], lgr5_positive_fraction * 100, width=0.05, align='edge', color=bin_colors)

    # Draw text near the top of the axes
    trans = matplotlib.transforms.blended_transform_factory(
        ax_lgr5.transData, ax_lgr5.transAxes)
    for i in range(len(bins) - 1):
        if lgr5_positive[i] + lgr5_negative[i] >= _MIN_COUNT_IN_BIN:
            ax_lgr5.text((bins[i] + bins[i + 1]) / 2, 1, str(lgr5_positive[i] + lgr5_negative[i]),
                         ha='center', va='top', fontsize=8, transform=trans)
    ax_lgr5.set_xlabel("Stem-to-enterocyte axis")
    ax_lgr5.set_ylabel("Positive cells (%)")

    ax_lgr5.set_xlim(0.78, 0.1)
    ax_lgr5.set_ylim(0, 85)

    # Hide top and right spines
    for ax in [ax_ki67_mean, ax_ki67_fraction, ax_lgr5]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
