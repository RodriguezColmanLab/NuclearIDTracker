from matplotlib import pyplot as plt
from typing import List

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


class _ProbabilitiesOverTime:
    time_h: List[float]
    stem_cell_counts: List[int]
    enterocyte_counts: List[int]
    paneth_cell_counts: List[float]

    def __init__(self):
        self.time_h = []
        self.stem_cell_counts = []
        self.enterocyte_counts = []
        self.paneth_cell_counts = []


def main():
    experiments = list_io.load_experiment_list_file(_DATA_FILE, load_images=False)

    for experiment in experiments:
        probabilities = _calculate_counts_over_time(experiment)

        figure = lib_figures.new_figure(size=(4, 3))
        figure.suptitle(experiment.name.get_name())
        ax = figure.gca()

        ax.set_ylim(0, max(max(probabilities.stem_cell_counts), max(probabilities.enterocyte_counts), max(probabilities.paneth_cell_counts)) * 1.1)
        ax.set_ylabel("# cells")
        ax.plot(probabilities.time_h, probabilities.stem_cell_counts, color=lib_figures.CELL_TYPE_PALETTE["STEM"], linewidth=3, label="Stem cell")
        ax.plot(probabilities.time_h, probabilities.enterocyte_counts, color=lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"], linewidth=3, label="Enterocyte")
        ax.plot(probabilities.time_h, probabilities.paneth_cell_counts, color=lib_figures.CELL_TYPE_PALETTE["PANETH"], linewidth=3, label="Paneth cell")
        ax.set_xlabel("Time (h)")

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

        figure.tight_layout()
        plt.show()


def _calculate_counts_over_time(experiment: Experiment) -> _ProbabilitiesOverTime:
    probabilities = _ProbabilitiesOverTime()
    timings = experiment.images.timings()

    for time_point in experiment.positions.time_points():
        stem_cell_count = 0
        enterocyte_count = 0
        paneth_cell_count = 0

        for position in experiment.positions.of_time_point(time_point):
            position_type = position_markers.get_position_type(experiment.position_data, position)

            if position_type == "STEM":
                stem_cell_count += 1
            elif position_type == "ENTEROCYTE":
                enterocyte_count += 1
            elif position_type == "PANETH":
                paneth_cell_count += 1

        probabilities.time_h.append(timings.get_time_h_since_start(time_point))
        probabilities.stem_cell_counts.append(stem_cell_count)
        probabilities.enterocyte_counts.append(enterocyte_count)
        probabilities.paneth_cell_counts.append(paneth_cell_count)

    return probabilities


if __name__ == "__main__":
    main()