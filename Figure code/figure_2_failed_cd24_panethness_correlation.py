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
_INTENSITY_KEY = "intensity_cd24"


def main():
    experiments = list(list_io.load_experiment_list_file(_EXPERIMENTS_FILE))

    cd24_values, panethness_values, organoid_indices = _get_intensity_against_paneth_chance_all(experiments)
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


def _get_intensity_against_paneth_chance_all(experiments: List[Experiment]) -> Tuple[List[float], List[float], List[int]]:
    all_cd24_values = list()
    all_panethness_values = list()
    all_organoid_indices = list()
    for i, experiment in enumerate(experiments):
        cd24_values, panethness_values = _get_intensity_against_paneth_chance_single_experiment(experiment)

        for cd24_value, panethness_value in zip(cd24_values, panethness_values):
            all_cd24_values.append(cd24_value)
            all_panethness_values.append(panethness_value)
            all_organoid_indices.append(i)
    return all_cd24_values, all_panethness_values, all_organoid_indices


def _get_intensity_against_paneth_chance_single_experiment(experiment: Experiment) -> Tuple[ndarray, ndarray]:
    paneth_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")
    cd24_values = list()
    panethness_values = list()

    for track in experiment.links.find_all_tracks():
        track_panethness_values = list()
        track_cd24_values = list()
        for position in track.positions():
            cell_probabilities = experiment.position_data.get_position_data(position,"ct_probabilities")
            if cell_probabilities is None:
                continue
            panethness = cell_probabilities[paneth_index]

            cd24_signal = intensity_calculator.get_raw_intensity(experiment.position_data, position, intensity_key=_INTENSITY_KEY)
            if cd24_signal is None:
                continue

            track_panethness_values.append(panethness)
            track_cd24_values.append(cd24_signal)
        if len(track_cd24_values) > 0:
            cd24_values.append(numpy.mean(track_cd24_values))
            panethness_values.append(numpy.mean(track_panethness_values))

    return numpy.array(cd24_values), numpy.array(panethness_values)


if __name__ == "__main__":
    main()
