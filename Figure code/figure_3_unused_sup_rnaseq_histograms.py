import numpy
import pandas
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures

_DATA_FILE = "../../Data/Serra2019 scRNAseq/TA_Stem score_all cells.csv"


def main():
    # Load the CSV file into a pandas DataFrame
    data = pandas.read_csv(_DATA_FILE)

    # There's a column named cell_ID, and the first two letters of each cell ID indicate the day
    # Extract this to a new column named day
    data["day"] = data["cell_ID"].str[:2]

    # Scale stem and EC from 0 to 1, clipping the extreme values
    stem_min = numpy.quantile(data["Stem_score"], 0.01)
    stem_max = numpy.quantile(data["Stem_score"], 0.99)
    ta_min = numpy.quantile(data["TA_score"], 0.01)
    ta_max = numpy.quantile(data["TA_score"], 0.99)
    ec_min = numpy.quantile(data["EC_score"], 0.01)
    ec_max = numpy.quantile(data["EC_score"], 0.99)

    # Do a Wilcoxon rank-sum test to compare the stem scores between day 3 and day 5
    stem_p_value = _wilcoxon_rank_sum_d3_d5(data, "Stem_score")
    ta_p_value = _wilcoxon_rank_sum_d3_d5(data, "TA_score")
    ec_p_value = _wilcoxon_rank_sum_d3_d5(data, "EC_score")

    # Make a histogram of the stem and EC scores
    figure = lib_figures.new_figure(size=(6, 1.7))
    ax_stem, ax_ta, ax_ec = figure.subplots(1, 3)
    for day in sorted(list(set(data["day"]))):
        _draw_histogram(ax_stem, data, "Proliferation_score", day)
        _draw_histogram(ax_ta, data, "TA_score", day)
        _draw_histogram(ax_ec, data, "EC_score", day)
    ax_stem.text(0.5, 1, f"p = {stem_p_value:.2}", transform=ax_stem.transAxes, ha="center", va="top")
    ax_ta.text(0.5, 1, f"p = {ta_p_value:.2}", transform=ax_ta.transAxes, ha="center", va="top")
    ax_ec.text(0.5, 1, f"p = {ec_p_value:.2}", transform=ax_ec.transAxes, ha="center", va="top")
    ax_stem.set_xlabel("Stem score")
    ax_ta.set_xlabel("TA score")
    ax_ec.set_xlabel("EC score")
    ax_ec.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax_stem.set_ylabel("Cell count")
    for ax in [ax_stem, ax_ta, ax_ec]:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    figure.tight_layout()
    plt.show()


def _draw_histogram(ax: Axes, data: pandas.DataFrame, score_name: str, day: str):
    """Draws the histogram of the given score for the given day on the given axis. DataFrame must have a "day" column
    and a score_name column."""
    color = "#636e72" if day == "d3" else "#f39c12"
    label = day.replace("d", "Day ")

    if day == "d5":
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

    score_min = numpy.quantile(data[score_name], 0.01)
    score_max = numpy.quantile(data[score_name], 0.99)
    ax.hist(data[score_name][data["day"] == day], bins=numpy.linspace(score_min, score_max, 25), alpha=0.7,
            label=label, color=color)


def _wilcoxon_rank_sum_d3_d5(data: pandas.DataFrame, column: str) -> float:
    stem_d3 = data[column][data["day"] == "d3"]
    stem_d5 = data[column][data["day"] == "d5"]
    _, stem_p_value = scipy.stats.ranksums(stem_d3, stem_d5)
    return stem_p_value


if __name__ == "__main__":
    main()
