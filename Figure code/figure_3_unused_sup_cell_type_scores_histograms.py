import numpy
import pandas
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
import lib_data
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

_DATA_FILE = "../../Data/Tracking data as controls/Dataset - full overweekend.autlist"
_TIME_POINT_H = [0, 60]


def _search_time_point(experiment: Experiment, desired_time_point_h: float) -> TimePoint:
    timings = experiment.images.timings()
    first_time_h = timings.get_time_h_since_start(experiment.positions.first_time_point())
    for time_point in experiment.positions.time_points():
        time_h = timings.get_time_h_since_start(time_point) - first_time_h
        if time_h >= desired_time_point_h:
            return time_point
    raise ValueError(f"Time point {desired_time_point_h} not found in experiment {experiment.name}")


def _build_dataframe() -> pandas.DataFrame:
    time_point_column = list()
    stemness_column = list()
    enterocyteness_column = list()
    stem_to_ec_column = list()

    for experiment in list_io.load_experiment_list_file(_DATA_FILE):
        position_data = experiment.position_data
        cell_types = experiment.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise ValueError(f"Experiment {experiment.name} does not have cell type probabilities")

        stem_index = cell_types.index("STEM")
        enterocyte_index = cell_types.index("ENTEROCYTE")
        for time_point_index, time_point_h in enumerate(_TIME_POINT_H):
            time_point = _search_time_point(experiment, time_point_h)
            for position in experiment.positions.of_time_point(time_point):
                ct_probabilities = position_data.get_position_data(position, "ct_probabilities")
                if ct_probabilities is None:
                    continue

                time_point_column.append(time_point_index)
                stemness_column.append(ct_probabilities[stem_index])
                enterocyteness_column.append(ct_probabilities[enterocyte_index])
                stem_to_ec_column.append(lib_data.find_stem_to_ec_location(cell_types, ct_probabilities))

    print(stem_to_ec_column)
    return pandas.DataFrame({
        "time_point_index": time_point_column,
        "stemness": stemness_column,
        "enterocyteness": enterocyteness_column,
        "stem_to_ec_location": stem_to_ec_column
    })


def main():
    data = _build_dataframe()

    # Do a Wilcoxon rank-sum test to compare the stem scores between day 3 and day 5
    stem_p_value = _wilcoxon_rank_sum_d3_d5(data, "stemness")
    ec_p_value = _wilcoxon_rank_sum_d3_d5(data, "enterocyteness")

    # Make a histogram of the stem and EC scores
    figure = lib_figures.new_figure(size=(6, 1.7))
    ax_stem, ax_stem_to_ec, ax_ec = figure.subplots(1, 3)
    for time_point_index in sorted(list(set(data["time_point_index"]))):
        _draw_histogram(ax_stem, data, "stemness", time_point_index)
        _draw_histogram(ax_stem_to_ec, data, "stem_to_ec_location", time_point_index)
        _draw_histogram(ax_ec, data, "enterocyteness", time_point_index)
    ax_stem.text(0.5, 1, f"p = {stem_p_value:.2}", transform=ax_stem.transAxes, ha="center", va="top")
    ax_ec.text(0.5, 1, f"p = {ec_p_value:.2}", transform=ax_ec.transAxes, ha="center", va="top")
    ax_stem.set_xlabel("Stem score")
    ax_stem_to_ec.set_xlabel("Stem to EC location")
    ax_stem_to_ec.invert_xaxis()
    ax_ec.set_xlabel("EC score")
    ax_ec.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax_stem.set_ylabel("Cell count")
    for ax in [ax_stem, ax_stem_to_ec, ax_ec]:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    figure.tight_layout()
    plt.show()


def _draw_histogram(ax: Axes, data: pandas.DataFrame, score_name: str, time_point_index: int):
    """Draws the histogram of the given score for the given day on the given axis. DataFrame must have a "day" column
    and a score_name column."""
    color = "#636e72" if time_point_index == 0 else "#f39c12"
    label = str(_TIME_POINT_H[time_point_index]) + "h"

    if time_point_index == 1:
        # Manually add histogram to the legend
        ax.add_artist(plt.Rectangle((0, 0), 0, 0, color=color, alpha=0.7, label=label))

        # And create the second y-axis
        ax = ax.twinx()
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_color(color)
    else:
        ax.spines["left"].set_color(color)

    # Set ax ytick color to match the histogram color
    ax.tick_params(axis='y', color=color, labelcolor=color)

    score_min = numpy.nanquantile(data[score_name], 0.01)
    score_max = numpy.nanquantile(data[score_name], 0.99)
    ax.hist(data[score_name][data["time_point_index"] == time_point_index], bins=numpy.linspace(score_min, score_max, 25), alpha=0.7,
            label=label, color=color)


def _wilcoxon_rank_sum_d3_d5(data: pandas.DataFrame, column: str) -> float:
    stem_d3 = data[column][data["time_point_index"] == 0]
    stem_d5 = data[column][data["time_point_index"] == 1]
    _, stem_p_value = scipy.stats.ranksums(stem_d3, stem_d5)
    return stem_p_value


if __name__ == "__main__":
    main()
