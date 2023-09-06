from enum import Enum, auto
from typing import Tuple, List

import numpy
import scipy
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numpy import ndarray

import lib_figures
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import intensity_calculator

_EXPERIMENTS_FILE = "../../Data/Testing data - output - CD24 and Lgr5.autlist"
_CD24_AVERAGING_WINDOW_H = 4
_TIME_OFFSET_H = 0
_INTENSITY_KEY = "intensity_cd24"


def main():
    experiments = list(list_io.load_experiment_list_file(_EXPERIMENTS_FILE))

    # offsets_h = list()
    # correlation_coefficient = list()
    # for offset_h in numpy.arange(-30, 30, 0.05):
    #     cd24_values, panethness_values = _get_intensity_against_paneth_chance(experiments, offset_h=offset_h)
    #     if len(cd24_values) < 40:
    #         continue
    #     result = scipy.stats.pearsonr(cd24_values, panethness_values)
    #
    #     offsets_h.append(offset_h)
    #     correlation_coefficient.append(result.statistic)

    # figure = lib_figures.new_figure()
    # ax = figure.gca()
    # ax.plot(offsets_h, correlation_coefficient, color="black")
    # ax.set_xlabel("Offset (h)")
    # ax.set_ylabel("Correlation coefficient")
    # plt.show()

    cd24_values, panethness_values, organoid_indices = _get_intensity_against_paneth_chance_all(experiments, offset_h=_TIME_OFFSET_H)
    result = scipy.stats.pearsonr(cd24_values, panethness_values, alternative="greater")
    figure = lib_figures.new_figure()
    ax = figure.gca()
    ax.scatter(cd24_values, panethness_values, s=20, c=organoid_indices, lw=0, cmap=LinearSegmentedColormap.from_list("custom", [
        "#d63031", "#00b894", "#0984e3"
    ]))
    ax.set_xlabel("CD24")
    ax.set_ylabel("Paneth likelihood")
    ax.set_title(f"R={result.statistic}, p={result.pvalue}")
    plt.show()


def _get_intensity_against_paneth_chance_all(experiments: List[Experiment], *,
                                             offset_h: float = 0) -> Tuple[List[float], List[float], List[int]]:
    all_cd24_values = list()
    all_panethness_values = list()
    all_organoid_indices = list()
    for i, experiment in enumerate(experiments):
        cd24_values, panethness_values = _get_intensity_against_paneth_chance_single_experiment(experiment, offset_h)
        cd24_values /= numpy.quantile(cd24_values, 0.9)

        for cd24_value, panethness_value in zip(cd24_values, panethness_values):
            all_cd24_values.append(cd24_value)
            all_panethness_values.append(panethness_value)
            all_organoid_indices.append(i)
    return all_cd24_values, all_panethness_values, all_organoid_indices


def _get_intensity_against_paneth_chance_single_experiment(experiment: Experiment, offset_h: float
                                                           ) -> Tuple[ndarray, ndarray]:
    paneth_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")
    cd24_values = list()
    panethness_values = list()

    resolution = experiment.images.resolution()
    time_point_divider = int(0.5 * _CD24_AVERAGING_WINDOW_H / resolution.time_point_interval_h)

    for time_point in experiment.positions.time_points():
        if time_point.time_point_number() % time_point_divider != 0:
            continue

        offset_time_point = TimePoint(
            time_point.time_point_number() - int(offset_h / resolution.time_point_interval_h))
        for position in experiment.positions.of_time_point(time_point):
            position_with_time_offset = experiment.links.get_position_near_time_point(position, offset_time_point)
            if position_with_time_offset.time_point() != offset_time_point:
                continue

            cd24_signal = intensity_calculator.get_normalized_intensity_over_time(experiment, position,
                                                                                  _CD24_AVERAGING_WINDOW_H,
                                                                                  intensity_key=_INTENSITY_KEY)
            if cd24_signal is None:
                continue
            cell_probabilities = experiment.position_data.get_position_data(position_with_time_offset,
                                                                            "ct_probabilities")
            if cell_probabilities is None:
                continue
            panethness = cell_probabilities[paneth_index]

            if panethness > 0.6 and cd24_signal.mean < 300:
                print(experiment.name, position, cd24_signal.mean, panethness)

            cd24_values.append(cd24_signal.mean)
            panethness_values.append(panethness)
    return numpy.array(cd24_values), numpy.array(panethness_values)


if __name__ == "__main__":
    main()
