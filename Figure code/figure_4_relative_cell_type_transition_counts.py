import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import figure_3_sup_cell_type_transition_matrix as cell_type_transition_panel
import lib_figures

_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


def main():
    transition_matrix_control = cell_type_transition_panel.calculate_transition_matrix(_DATA_FILE_CONTROL)
    transition_matrix_regeneration = cell_type_transition_panel.calculate_transition_matrix(_DATA_FILE_REGENERATION)

    figure = lib_figures.new_figure(size=(3, 4.5))
    ax_control, ax_regeneration = figure.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    _plot_bars_relative(ax_control, transition_matrix_control)
    _plot_bars_relative(ax_regeneration, transition_matrix_regeneration)

    ax_control.set_title("Control")
    ax_regeneration.set_title("Regeneration")

    bar_x_positions = numpy.arange(cell_type_transition_panel.STEM_TO_ENTEROCYTE_BINS + 1) - 0.5
    bar_x_labels = [(f"{i/cell_type_transition_panel.STEM_TO_ENTEROCYTE_BINS}" if i % 2 == 0 else "") for i in range(cell_type_transition_panel.STEM_TO_ENTEROCYTE_BINS + 1)]
    ax_regeneration.set_xticks(bar_x_positions)
    ax_regeneration.set_xticklabels(bar_x_labels)

    ax_control.legend()
    ax_regeneration.set_xlim(cell_type_transition_panel.STEM_TO_ENTEROCYTE_BINS - 0.5, -0.5, )
    plt.show()


def _plot_bars_absolute(ax: Axes, transition_matrix: cell_type_transition_panel.CellTypeTransitionMatrix):
    bins = cell_type_transition_panel.STEM_TO_ENTEROCYTE_BINS
    count_matrix = transition_matrix.count_matrix_stem_to_ec
    bars_stayed_same = [count_matrix[i, i] for i in range(bins)]
    bars_increased = [sum(count_matrix[i, i:]) for i in range(bins)]
    bars_decreased = [sum(count_matrix[i, :i]) for i in range(bins)]

    ax.bar(range(bins), bars_increased, color="#00ff00", label="Dedifferentation",
           bottom=(numpy.array(bars_decreased) + bars_stayed_same))
    ax.bar(range(bins), bars_stayed_same, color="#eeeeee", label="No change", bottom=bars_decreased)
    ax.bar(range(bins), bars_decreased, color="#0000ff", label="Differentation")
    ax.set_ylabel("Cell count")


def _plot_bars_relative(ax: Axes, transition_matrix: cell_type_transition_panel.CellTypeTransitionMatrix):
    bins = cell_type_transition_panel.STEM_TO_ENTEROCYTE_BINS
    count_matrix = transition_matrix.count_matrix_stem_to_ec
    bars_stayed_same = [count_matrix[i, i] for i in range(bins)]
    bars_increased = [sum(count_matrix[i, i:]) for i in range(bins)]
    bars_decreased = [sum(count_matrix[i, :i]) for i in range(bins)]
    bars_total = numpy.array(bars_stayed_same) + numpy.array(bars_increased) + numpy.array(bars_decreased)

    print("Bars left edges:", [bin / cell_type_transition_panel.STEM_TO_ENTEROCYTE_BINS for bin in  range(bins)])
    print("Stayed the same:", bars_stayed_same)
    print("Dedifferentation:", bars_increased)
    print("Differentiation:", bars_decreased)

    ax.bar(range(bins), bars_increased / bars_total, color="#00ff00", label="Dedifferentation",
           bottom=(numpy.array(bars_decreased) + bars_stayed_same) / bars_total)
    ax.bar(range(bins), bars_stayed_same / bars_total, color="#eeeeee", label="No change", bottom=bars_decreased / bars_total)
    ax.bar(range(bins), bars_decreased / bars_total, color="#0000ff", label="Differentation")

    # Add the total count
    for i, count in enumerate(bars_total):
        if count == 0:
            continue
        color = "black" if (bars_decreased[i] / bars_total[i]) < 0.95 else "white"
        ax.text(i, 0.98, str(int(count)), horizontalalignment="center", verticalalignment="top", color=color)
    ax.set_ylabel("Cell fraction")


if __name__ == "__main__":
    main()