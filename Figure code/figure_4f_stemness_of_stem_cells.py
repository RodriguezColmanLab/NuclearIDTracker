from typing import Optional, List, Union, Any, Dict

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.patches import Arc
from numpy import ndarray

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_AVERAGING_WINDOW_WIDTH_H = 5
_DATASET_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATASET_FILE_ABLATION = "../../Data/Stem cell regeneration/Dataset - during DT treatment.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


def _find_cell_type(position_data: PositionData, track: LinkingTrack) -> Optional[str]:
    """Finds the most common cell type in the track. Returns None if no cell type is found at all in the track."""
    cell_type_counts = dict()
    for position in track.positions():
        cell_type = position_markers.get_position_type(position_data, position)
        if cell_type is not None:
            cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
    if len(cell_type_counts) == 0:
        return None
    return max(cell_type_counts, key=cell_type_counts.get)


def _plot_organoid_averages(ax: Axes, data: Dict[str, List[float]], x: float):
    means = [numpy.mean(data_points) for data_points in data.values()]
    random = numpy.random.Generator(numpy.random.MT19937(seed=int(x)))
    x_positions = random.normal(x, 0.06, size=len(means))
    ax.scatter(x_positions, means, color="black", s=10, marker="s", lw=0)


def main():
    # Collect regeneration data
    regeneration_start_data_by_experiment = dict()
    regeneration_end_data_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_REGENERATION):
        regeneration_start_data_by_experiment[experiment.name.get_name()] = _extract_stemness(experiment)
        regeneration_end_data_by_experiment[experiment.name.get_name()] = _extract_stemness(experiment, at_start=False)

    # Collect ablation data
    ablation_data_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_ABLATION):
        ablation_data_by_experiment[experiment.name.get_name()] = _extract_stemness(experiment)

    # Collect control data
    control_data_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_CONTROL):
        control_data_by_experiment[experiment.name.get_name()] = _extract_stemness(experiment)

    # Sum up the data for each group
    control_summed_data = sum(control_data_by_experiment.values(), [])
    ablation_summed_data = sum(ablation_data_by_experiment.values(), [])
    regeneration_start_summed_data = sum(regeneration_start_data_by_experiment.values(), [])
    regeneration_end_summed_data = sum(regeneration_end_data_by_experiment.values(), [])

    figure = lib_figures.new_figure(size=(5, 4))
    ax = figure.gca()
    ax.set_ylim(0.2, 0.7)
    violin = ax.violinplot([control_summed_data, ablation_summed_data, regeneration_start_summed_data, regeneration_end_summed_data],
                  showmeans=False, showextrema=False, showmedians=False, widths=0.75)
    for body in violin['bodies']:
        body.set_facecolor(lib_figures.CELL_TYPE_PALETTE["STEM"])
        body.set_alpha(0.9)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Control", "Ablation start", "Regeneration start", "+24 h"])
    _plot_organoid_averages(ax, control_data_by_experiment, 1)
    _plot_organoid_averages(ax, ablation_data_by_experiment, 2)
    _plot_organoid_averages(ax, regeneration_start_data_by_experiment, 3)
    _plot_organoid_averages(ax, regeneration_end_data_by_experiment, 4)
    plt.show()


def _extract_stemness(experiment: Experiment, *, at_start: bool = True) -> List[float]:
    """Gets the stemness probabilities of all positions in tracks predicted as stem cells."""
    cell_types = experiment.global_data.get_data("ct_probabilities")
    stemness = list()
    stem_cell_index = cell_types.index("STEM")

    time_point = experiment.positions.first_time_point() if at_start else experiment.positions.last_time_point()
    for position in experiment.positions.of_time_point(time_point):

        ct_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
        if ct_probabilities is None:
            continue

        if numpy.argmax(ct_probabilities) == stem_cell_index:
            stemness.append(ct_probabilities[stem_cell_index])

    return stemness


def _extract_stemness_(experiment: Experiment) -> List[float]:
    """Gets the stemness probabilities of all positions in tracks predicted as stem cells."""
    cell_types = experiment.global_data.get_data("ct_probabilities")
    stemness = list()
    stem_cell_index = cell_types.index("STEM")

    for track in experiment.links.find_all_tracks():
        cell_type = _find_cell_type(experiment.position_data, track)
        if cell_type != "STEM":
            continue

        for position in track.positions():
            ct_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
            if ct_probabilities is None:
                continue
            stemness.append(ct_probabilities[stem_cell_index])

    return stemness


if __name__ == "__main__":
    main()