from collections import defaultdict
from typing import List, NamedTuple, Dict, Callable

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

_DATA_FILE = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"

_BLOCK_SIZE_H = 1


class _ProbabilitiesInTimeBlock:
    time_h: float
    stem_cell_probabilities: List[float]
    enterocyte_probabilities: List[float]
    paneth_cell_probabilities: List[float]

    def __init__(self, time_h: float):
        self.time_h = time_h
        self.stem_cell_probabilities = list()
        self.enterocyte_probabilities = list()
        self.paneth_cell_probabilities = list()


class _ProbabilitiesOverTime:
    _time_blocks: Dict[int, _ProbabilitiesInTimeBlock]

    def __init__(self):
        self._time_blocks = dict()

    def get_time_block(self, time_h: float) -> _ProbabilitiesInTimeBlock:
        """Returns the time block for the given time point. If the time block does not exist, it is created."""
        time_index = int(time_h / _BLOCK_SIZE_H)
        if time_index not in self._time_blocks:
            self._time_blocks[time_index] = _ProbabilitiesInTimeBlock(time_index * _BLOCK_SIZE_H)
        return self._time_blocks[time_index]

    def get_block_times(self) -> List[float]:
        """Returns the time points of the time blocks in ascending order."""
        return [key * _BLOCK_SIZE_H for key in sorted(self._time_blocks.keys())]


def _plot_histograms_over_time(ax: Axes, probabilities_over_time: _ProbabilitiesOverTime,
                               block_getter: Callable[[_ProbabilitiesInTimeBlock], List[float]]):
    scale_factor = 15 / _BLOCK_SIZE_H
    times_h = probabilities_over_time.get_block_times()

    for time_h in times_h:
        block = probabilities_over_time.get_time_block(time_h)

        # Create a vertical histogram of the probabilities
        probabilities = block_getter(block)
        ax.hist(probabilities, bins="scott", orientation='horizontal', color="#4152A3", bottom=time_h * scale_factor, density=True)

    ax.set_xticks([time_h * scale_factor for time_h in times_h])
    ax.set_xticklabels([f"{time_h:.0f} h" for time_h in times_h], ha="left")


def main():
    experiments = list_io.load_experiment_list_file(_DATA_FILE)

    for experiment in experiments:
        probabilities = _calculate_probabilities_over_time(experiment)

        figure = lib_figures.new_figure(size=(11, 3))
        figure.suptitle(experiment.name.get_name())
        axes = figure.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

        _plot_histograms_over_time(axes[0], probabilities, lambda block: block.stem_cell_probabilities)
        axes[0].set_title("Stem cell")
        axes[0].set_xlim(-4, 24)
        axes[0].set_ylim(0, 0.8)
        axes[0].set_ylabel("Probability")

        _plot_histograms_over_time(axes[1], probabilities, lambda block: block.enterocyte_probabilities)
        axes[1].set_title("Enterocyte")
        axes[1].set_xlabel("Time (h)")

        _plot_histograms_over_time(axes[2], probabilities, lambda block: block.paneth_cell_probabilities)
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

        time_block = probabilities.get_time_block(timings.get_time_h_since_start(time_point))
        time_block.stem_cell_probabilities.extend(stem_cell_probabilities)
        time_block.enterocyte_probabilities.extend(enterocyte_probabilities)
        time_block.paneth_cell_probabilities.extend(paneth_cell_probabilities)

    return probabilities


if __name__ == "__main__":
    main()