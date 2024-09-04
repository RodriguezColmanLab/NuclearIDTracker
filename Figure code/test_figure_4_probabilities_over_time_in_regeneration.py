from typing import List, NamedTuple

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

_DATA_FILE = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


class _ProbabilitiesOverTime:
    time_h: List[float]
    stem_cell_means: List[float]
    stem_cell_stds: List[float]
    enterocyte_means: List[float]
    enterocyte_stds: List[float]
    paneth_cell_means: List[float]
    paneth_cell_stds: List[float]

    def __init__(self):
        self.time_h = []
        self.stem_cell_means = []
        self.stem_cell_stds = []
        self.enterocyte_means = []
        self.enterocyte_stds = []
        self.paneth_cell_means = []
        self.paneth_cell_stds = []


def _plot_means_and_stds(ax: Axes, time_h: List[float], means: List[float], stds: List[float]):
    ax.plot(time_h, means)
    ax.fill_between(time_h,
                    numpy.array(means) - numpy.array(stds),
                    numpy.array(means) + numpy.array(stds),
                    alpha=0.5)


def main():
    experiments = list_io.load_experiment_list_file(_DATA_FILE)

    for experiment in experiments:
        probabilities = _calculate_probabilities_over_time(experiment)

        figure = lib_figures.new_figure(size=(8, 3))
        figure.suptitle(experiment.name.get_name())
        axes = figure.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

        _plot_means_and_stds(axes[0], probabilities.time_h, probabilities.stem_cell_means, probabilities.stem_cell_stds)
        axes[0].set_title("Stem cell")
        axes[0].set_ylim(0, 0.5)
        axes[0].set_ylabel("Probability")

        _plot_means_and_stds(axes[1], probabilities.time_h, probabilities.enterocyte_means, probabilities.enterocyte_stds)
        axes[1].set_title("Enterocyte")
        axes[1].set_xlabel("Time (h)")

        _plot_means_and_stds(axes[2], probabilities.time_h, probabilities.paneth_cell_means, probabilities.paneth_cell_stds)
        axes[2].set_title("Paneth cell")

        figure.tight_layout()
        plt.show()


def _calculate_probabilities_over_time(experiment: Experiment) -> _ProbabilitiesOverTime:
    probabilities = _ProbabilitiesOverTime()
    timings = experiment.images.timings()

    stem_cell_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
    enterocyte_index = experiment.global_data.get_data("ct_probabilities").index("ENTEROCYTE")
    paneth_cell_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")

    for time_point in experiment.positions.time_points():
        stem_cell_probabilities = []
        enterocyte_probabilities = []
        paneth_cell_probabilities = []

        for position in experiment.positions.of_time_point(time_point):
            position_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
            if position_probabilities is None:
                continue

            stem_cell_probabilities.append(position_probabilities[stem_cell_index])
            enterocyte_probabilities.append(position_probabilities[enterocyte_index])
            paneth_cell_probabilities.append(position_probabilities[paneth_cell_index])

        probabilities.time_h.append(timings.get_time_h_since_start(time_point))
        probabilities.stem_cell_means.append(sum(stem_cell_probabilities) / len(stem_cell_probabilities))
        probabilities.stem_cell_stds.append(numpy.std(stem_cell_probabilities, ddof=1))
        probabilities.enterocyte_means.append(sum(enterocyte_probabilities) / len(enterocyte_probabilities))
        probabilities.enterocyte_stds.append(numpy.std(enterocyte_probabilities, ddof=1))
        probabilities.paneth_cell_means.append(sum(paneth_cell_probabilities) / len(paneth_cell_probabilities))
        probabilities.paneth_cell_stds.append(numpy.std(paneth_cell_probabilities, ddof=1))

    return probabilities


if __name__ == "__main__":
    main()