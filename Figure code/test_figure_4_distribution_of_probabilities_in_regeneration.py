from typing import List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
from organoid_tracker.imaging import list_io

_DATA_FILE = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


def _plot_distribution(ax: Axes, all_probablities: List[float], name: str):
    ax.hist(all_probablities, bins="scott")
    ax.set_title(name)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")


def main():
    experiments = list_io.load_experiment_list_file(_DATA_FILE)

    stem_cell_probabilities = []
    enterocyte_probabilities = []
    paneth_cell_probabilities = []
    for experiment in experiments:
        stem_cell_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
        enterocyte_index = experiment.global_data.get_data("ct_probabilities").index("ENTEROCYTE")
        paneth_cell_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")

        for position, probabilities in experiment.position_data.find_all_positions_with_data("ct_probabilities"):
            stem_cell_probabilities.append(probabilities[stem_cell_index])
            enterocyte_probabilities.append(probabilities[enterocyte_index])
            paneth_cell_probabilities.append(probabilities[paneth_cell_index])


    figure = lib_figures.new_figure()
    axes = figure.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    _plot_distribution(axes[0], stem_cell_probabilities, "Stem cell")
    _plot_distribution(axes[1], enterocyte_probabilities, "Enterocyte")
    _plot_distribution(axes[2], paneth_cell_probabilities, "Paneth cell")

    plt.show()


if __name__ == "__main__":
    main()