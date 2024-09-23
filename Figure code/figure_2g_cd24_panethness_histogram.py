import numpy
from matplotlib import pyplot as plt

import lib_figures
from organoid_tracker.imaging import list_io

_EXPERIMENTS_FILE = "../../Data/Testing data - output - CD24 and Lgr5.autlist"


def main():
    panethness_negatives = []
    panethness_positives = []
    for experiment in list_io.load_experiment_list_file(_EXPERIMENTS_FILE):
        paneth_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")

        for time_point in experiment.time_points():
            for position in experiment.positions.of_time_point(time_point):
                probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
                if not probabilities:
                    continue
                panethness = probabilities[paneth_index]

                cd24_positive = experiment.position_data.get_position_data(position, "cd24_positive")
                if cd24_positive is None:
                    cd24_positive = False

                if cd24_positive:
                    panethness_positives.append(panethness)
                else:
                    panethness_negatives.append(panethness)

    figure = lib_figures.new_figure()
    ax = figure.gca()
    histogram_negatives, bins_negatives = numpy.histogram(panethness_negatives, bins=50)
    histogram_negatives = histogram_negatives / numpy.max(histogram_negatives)
    histogram_positives, bins_positives = numpy.histogram(panethness_positives, bins=bins_negatives)
    histogram_positives = histogram_positives / numpy.max(histogram_positives)

    ax.bar(bins_negatives[:-1], histogram_negatives, width=bins_negatives[1] - bins_negatives[0], alpha=0.5, label="CD24 negatives")
    ax.bar(bins_positives[:-1], histogram_positives, width=bins_positives[1] - bins_positives[0], alpha=0.5, label="CD24 positives")
    ax.legend()
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Predicted Paneth likelihood")
    ax.set_ylabel("Relative frequency")
    plt.show()


if __name__ == "__main__":
    main()
