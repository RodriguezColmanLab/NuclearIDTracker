from enum import Enum, auto
from typing import Tuple, List

import matplotlib.colors
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
_CHANCE_BINS = 50
_SIGNAL_BINS = 50
_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list("OurOranges", ["#FDB87C", "#812804"])


def main():
    experiments = list(list_io.load_experiment_list_file(_EXPERIMENTS_FILE))

    image = _get_intensity_against_paneth_chance_all(experiments, offset_h=_TIME_OFFSET_H)
    image = image.astype(numpy.float32)
    image[image == 0] = numpy.nan
    figure = lib_figures.new_figure()
    ax = figure.gca()
    plotted_image = ax.imshow(image, extent=(0, 1, 1, 0), cmap=_COLORMAP, vmin=0)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("CD24")
    ax.set_ylabel("Paneth likelihood")
    plt.colorbar(plotted_image).set_label("Count")
    plt.show()


def _get_intensity_against_paneth_chance_all(experiments: List[Experiment], *,
                                             offset_h: float = 0) -> ndarray:
    image = numpy.zeros((_CHANCE_BINS, _SIGNAL_BINS), dtype=numpy.uint32)
    for experiment in experiments:
        cd24_values, panethness_values = _get_intensity_against_paneth_chance_single_experiment(experiment, offset_h)
        cd24_min = numpy.min(cd24_values)
        cd24_max = numpy.max(cd24_values)
        for cd24_value, panethness_value in zip(cd24_values, panethness_values):
            cd24_value = (cd24_value - cd24_min) / (cd24_max - cd24_min)
            if cd24_value < 0:
                cd24_value = 0
            if cd24_value >= 1:
                cd24_value = 0.9999
            image[int(panethness_value * _CHANCE_BINS), int(cd24_value * _SIGNAL_BINS)] += 1
    return image


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

            cd24_values.append(cd24_signal.mean)
            panethness_values.append(panethness)
    return numpy.array(cd24_values), numpy.array(panethness_values)


if __name__ == "__main__":
    main()
