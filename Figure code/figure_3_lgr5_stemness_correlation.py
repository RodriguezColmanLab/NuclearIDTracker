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
_LGR5_AVERAGING_WINDOW_H = 4
_INTENSITY_KEY = "intensity_lgr5"


def main():
    experiments = list(list_io.load_experiment_list_file(_EXPERIMENTS_FILE))

    # offsets_h = list()
    # correlation_coefficient = list()
    # for offset_h in numpy.arange(-30, 30, 0.05):
    #     lgr5_values, stemness_values = _get_intensity_against_stem_chance(experiments, offset_h=offset_h)
    #     if len(lgr5_values) < 40:
    #         continue
    #     result = scipy.stats.pearsonr(lgr5_values, stemness_values)
    #
    #     offsets_h.append(offset_h)
    #     correlation_coefficient.append(result.statistic)

    # figure = lib_figures.new_figure()
    # ax = figure.gca()
    # ax.plot(offsets_h, correlation_coefficient, color="black")
    # ax.set_xlabel("Offset (h)")
    # ax.set_ylabel("Correlation coefficient")
    # plt.show()

    lgr5_values, stemness_values, organoid_indices = _get_intensity_against_stem_chance_all(experiments)
    result = scipy.stats.pearsonr(lgr5_values, stemness_values, alternative="greater")
    figure = lib_figures.new_figure()
    ax = figure.gca()
    ax.scatter(lgr5_values, stemness_values, s=20, c=organoid_indices, lw=0, cmap=LinearSegmentedColormap.from_list("custom", [
        "#d63031", "#00b894", "#0984e3"
    ]))
    ax.set_xlabel("Lgr5")
    ax.set_ylabel("Stem likelihood")
    ax.set_title(f"R={result.statistic}, p={result.pvalue}")
    plt.show()


def _get_intensity_against_stem_chance_all(experiments: List[Experiment]) -> Tuple[List[float], List[float], List[int]]:
    all_lgr5_values = list()
    all_stemness_values = list()
    all_organoid_indices = list()
    for i, experiment in enumerate(experiments):
        lgr5_values, stemness_values = _get_intensity_against_stem_chance_single_experiment(experiment)
        lgr5_values /= numpy.quantile(lgr5_values, 0.9)

        for lgr5_value, stemness_value in zip(lgr5_values, stemness_values):
            all_lgr5_values.append(lgr5_value)
            all_stemness_values.append(stemness_value)
            all_organoid_indices.append(i)
    return all_lgr5_values, all_stemness_values, all_organoid_indices


def _get_intensity_against_stem_chance_single_experiment(experiment: Experiment) -> Tuple[ndarray, ndarray]:
    stem_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
    lgr5_values = list()
    stemness_values = list()

    resolution = experiment.images.resolution()
    time_point_divider = int(0.5 * _LGR5_AVERAGING_WINDOW_H / resolution.time_point_interval_h) + 1

    time_point = TimePoint(experiment.positions.first_time_point_number() + time_point_divider)

    for position in experiment.positions.of_time_point(time_point):
        lgr5_signal = intensity_calculator.get_normalized_intensity_over_time(experiment, position,
                                                                              _LGR5_AVERAGING_WINDOW_H,
                                                                              intensity_key=_INTENSITY_KEY)
        if lgr5_signal is None:
            continue
        cell_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
        if cell_probabilities is None:
            continue
        stemness = cell_probabilities[stem_index]

        lgr5_values.append(lgr5_signal.mean)
        stemness_values.append(stemness)
    return numpy.array(lgr5_values), numpy.array(stemness_values)


if __name__ == "__main__":
    main()
