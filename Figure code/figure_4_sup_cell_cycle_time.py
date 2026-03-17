import matplotlib.colors
import numpy.random
from matplotlib import pyplot as plt
from typing import List, Dict, Any, Union

from matplotlib.axes import Axes
from scipy.stats import ttest_ind

from organoid_tracker.core import MPLColor
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io, io

import lib_figures

_DATASET_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


def _color_violin(violin: Dict[str, Any], color: MPLColor, average_bar_color: MPLColor = "black"):
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(1)
    if "cmeans" in violin:
        violin["cmeans"].set_color(average_bar_color)
    if "cmedians" in violin:
        violin["cmedians"].set_color(average_bar_color)


def main():
    control_cell_cycle_times = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_CONTROL):
        cell_cycle_times = _get_cell_cycle_times_h(experiment)
        control_cell_cycle_times[experiment.name.get_name()] = cell_cycle_times

    regeneration_cell_cycle_times = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_REGENERATION):
        cell_cycle_times = _get_cell_cycle_times_h(experiment)
        regeneration_cell_cycle_times[experiment.name.get_name()] = cell_cycle_times

    _plot_stacked_bars(control_cell_cycle_times, regeneration_cell_cycle_times)
    _plot_stacked_bars_per_organoid(control_cell_cycle_times, regeneration_cell_cycle_times)
    _plot_violins(control_cell_cycle_times, regeneration_cell_cycle_times)


def _calculate_bin_fractions(cell_cycle_times: Union[Dict[str, List[float]], List[float]], bins: List[float]) -> List[float]:
    """Calculate the fraction of cell cycle times that fall into each bin."""

    # Collect all cell cycle times into a single list
    if isinstance(cell_cycle_times, dict):
        all_cell_cycle_times = []
        for times in cell_cycle_times.values():
            all_cell_cycle_times.extend(times)
    else:
        all_cell_cycle_times = cell_cycle_times

    # Distribute cell cycle times into bins
    bin_counts = [0] * len(bins)
    for cell_cycle_time in all_cell_cycle_times:
        for i in range(len(bins) - 1, -1, -1):
            bin_start = bins[i]
            if cell_cycle_time >= bin_start:
                bin_counts[i] += 1
                break

    # Calculate fractions
    return [count / len(all_cell_cycle_times) for count in bin_counts]



def _plot_stacked_bars(control_cell_cycle_times: Dict[str, List[float]], regeneration_cell_cycle_times: Dict[str, List[float]]):
    figure = lib_figures.new_figure(size=(2.5, 3))
    ax = figure.gca()

    bins = [5, 10, 12]
    bin_fraction_control = _calculate_bin_fractions(control_cell_cycle_times, bins)
    bin_fraction_regeneration = _calculate_bin_fractions(regeneration_cell_cycle_times, bins)

    bin_labels = []
    for i in range(len(bins)):
        if i == len(bins) - 1:
            bin_labels.append(f"{bins[i]}+ h")
        elif i == 0:
            bin_labels.append(f"<{bins[i + 1]} h")
        else:
            bin_labels.append(f"{bins[i]} - {bins[i + 1]} h")

    _plot_stacked_bar(ax, 0, bin_fraction_control, bin_labels, color="#56BCFF")
    _plot_stacked_bar(ax, 1, bin_fraction_regeneration, bin_labels, color="#da5855")
    ax.set_xticks([0, 1], labels=["Control", "Regeneration"])
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    figure.tight_layout()
    plt.show()


def _plot_stacked_bars_per_organoid(control_cell_cycle_times: Dict[str, List[float]], regeneration_cell_cycle_times: Dict[str, List[float]]):
    figure = lib_figures.new_figure(size=(6, 3))
    ax = figure.gca()

    bins = [5, 8, 10, 12]

    bin_labels = []
    for i in range(len(bins)):
        if i == len(bins) - 1:
            bin_labels.append(f"{bins[i]}+ h")
        else:
            bin_labels.append(f"{bins[i]} - {bins[i+1]} h")

    x_positions = []
    labels = []
    x = 0
    for organoid_name, cell_cycle_times in control_cell_cycle_times.items():
        bin_fraction_control = _calculate_bin_fractions(cell_cycle_times, bins)
        _plot_stacked_bar(ax, x, bin_fraction_control, bin_labels, color="#56BCFF")

        x_positions.append(x)
        labels.append(f"{organoid_name} (C)")
        x += 1
    for organoid_name, cell_cycle_times in regeneration_cell_cycle_times.items():
        bin_fraction_regeneration = _calculate_bin_fractions(cell_cycle_times, bins)
        _plot_stacked_bar(ax, x, bin_fraction_regeneration, bin_labels, color="#da5855")

        x_positions.append(x)
        labels.append(f"{organoid_name} (R)")
        x += 1
    ax.set_xticks(x_positions, labels=labels, rotation=-45, ha="left")
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    figure.tight_layout()
    plt.show()


def _plot_stacked_bar(ax: Axes, x_position: float, bin_fractions: List[float], bin_labels: List[str], *, color: str = "#eeeeee"):
    bottom = 0
    for i in range(len(bin_fractions)):
        bin_fraction = bin_fractions[i]

        if bin_fraction > 0:
            text = bin_labels[i]
            ax.bar([x_position], [bin_fraction], bottom=[bottom], color=color)
            ax.text(x_position, bottom + bin_fraction / 2, text, ha="center", va="center", color="white", fontsize=8)

            bottom += bin_fraction

        # Make the color 20% darker for the next bin
        color = matplotlib.colors.to_rgb(color)
        color = tuple(max(c * 0.8, 0) for c in color)



def _plot_violins(control_cell_cycle_times: Dict[str, List[float]], regeneration_cell_cycle_times: Dict[str, List[float]]):
    # Plotting the data
    figure = lib_figures.new_figure()
    ax = figure.gca()
    x = 0

    labels_text = []
    labels_x = []
    max_cell_cycle_time = 0

    # Plot the control
    mean_cell_cycle_times_control = list()
    random = numpy.random.Generator(numpy.random.MT19937(seed=1))
    for experiment_name, cell_cycle_times in control_cell_cycle_times.items():
        ax.scatter(random.normal(loc=x, scale=0.07, size=len(cell_cycle_times)), cell_cycle_times, c="black", s=4,
                   marker="s", alpha=0.6, linewidths=0, zorder=10)
        _color_violin(ax.violinplot([cell_cycle_times], positions=[x], widths=0.7, showextrema=False, showmeans=False,
                                    showmedians=True), color="#b2bec3")
        mean_cell_cycle_times_control.append(sum(cell_cycle_times) / len(cell_cycle_times))
        max_cell_cycle_time = max(max_cell_cycle_time, max(cell_cycle_times))
        labels_text.append(experiment_name)
        labels_x.append(x)
        x += 1

    # Plot the regeneration
    x += 1
    mean_cell_cycle_times_regeneration = list()
    for experiment_name, cell_cycle_times in regeneration_cell_cycle_times.items():
        ax.scatter(random.normal(loc=x, scale=0.07, size=len(cell_cycle_times)), cell_cycle_times, c="black", s=4,
                   marker="s", alpha=0.6, linewidths=0, zorder=10)
        _color_violin(ax.violinplot([cell_cycle_times], positions=[x], widths=0.7, showextrema=False, showmeans=False,
                                    showmedians=True), color="#da5855")
        mean_cell_cycle_times_regeneration.append(sum(cell_cycle_times) / len(cell_cycle_times))
        max_cell_cycle_time = max(max_cell_cycle_time, max(cell_cycle_times))
        labels_text.append(experiment_name)
        labels_x.append(x)
        x += 1

    # Calculate the p-value between cell_cycle_times_control and cell_cycle_times_regeneration using scipy.stats.ttest_ind
    t_statistic, p_value = ttest_ind(mean_cell_cycle_times_control, mean_cell_cycle_times_regeneration)
    ax.text(x / 2, 3, f"p = {p_value:.2f}", ha="center", va="bottom")

    ax.set_xticks(labels_x)
    ax.set_xticklabels(labels_text, rotation=-45, ha="left")
    ax.set_ylabel("Cell cycle time (h)")
    ax.set_ylim(0, max_cell_cycle_time * 1.1)
    figure.tight_layout()
    plt.show()


def _get_cell_cycle_times_h(experiment: Experiment) -> List[float]:
    timings = experiment.images.timings()
    cell_cycle_times_h = []
    for track in experiment.links.find_all_tracks():
        previous_tracks = track.get_previous_tracks()
        if len(previous_tracks) != 1:
            continue
        if not previous_tracks.pop().will_divide():
            continue
        if not track.will_divide():
            continue

        # At this point, we know we captured a full cell cycle
        previous_division_time_point = track.first_time_point() - 1
        next_division_time_point = track.last_time_point()
        previous_division_time_h = timings.get_time_h_since_start(previous_division_time_point)
        next_division_time_h = timings.get_time_h_since_start(next_division_time_point)
        cell_cycle_time_h = next_division_time_h - previous_division_time_h

        if cell_cycle_time_h < 8:
            print("Short cell cycle time detected: " + str(cell_cycle_time_h) + " h in track " + experiment.name.get_name() + " / " + str(track.find_first_position()))

        cell_cycle_times_h.append(cell_cycle_time_h)

    return cell_cycle_times_h


if __name__ == "__main__":
    main()
