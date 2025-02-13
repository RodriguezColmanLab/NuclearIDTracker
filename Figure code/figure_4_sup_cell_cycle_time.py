import numpy.random
from matplotlib import pyplot as plt
from typing import List, Dict, Any
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
        ax.scatter(random.normal(loc=x, scale=0.07, size=len(cell_cycle_times)), cell_cycle_times, c="black", s=4, marker="s", alpha=0.6, linewidths=0, zorder=10)
        _color_violin(ax.violinplot([cell_cycle_times], positions=[x], widths=0.7, showextrema=False, showmeans=False, showmedians=True), color="#b2bec3")
        mean_cell_cycle_times_control.append(sum(cell_cycle_times) / len(cell_cycle_times))
        max_cell_cycle_time = max(max_cell_cycle_time, max(cell_cycle_times))
        labels_text.append(experiment_name)
        labels_x.append(x)
        x += 1

    # Plot the regeneration
    x += 1
    mean_cell_cycle_times_regeneration = list()
    for experiment_name, cell_cycle_times in regeneration_cell_cycle_times.items():
        ax.scatter(random.normal(loc=x, scale=0.07, size=len(cell_cycle_times)), cell_cycle_times, c="black", s=4, marker="s", alpha=0.6, linewidths=0, zorder=10)
        _color_violin(ax.violinplot([cell_cycle_times], positions=[x], widths=0.7, showextrema=False, showmeans=False, showmedians=True), color="#da5855")
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
        cell_cycle_times_h.append(cell_cycle_time_h)

    return cell_cycle_times_h


if __name__ == "__main__":
    main()
