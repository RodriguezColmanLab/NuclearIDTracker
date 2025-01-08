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
    stemness_regeneration_start_by_experiment = dict()
    stemness_regeneration_end_by_experiment = dict()
    enterocyteness_regeneration_start_by_experiment = dict()
    enterocyteness_regeneration_end_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_REGENERATION):
        stemness_regeneration_start_by_experiment[experiment.name.get_name()] \
            = _extract_probabilities_of_stem_cells(experiment, cell_type="STEM")
        enterocyteness_regeneration_start_by_experiment[experiment.name.get_name()] \
            = _extract_probabilities_of_stem_cells(experiment, cell_type="ENTEROCYTE")
        stemness_regeneration_end_by_experiment[experiment.name.get_name()] \
            = _extract_probabilities_of_stem_cells(experiment, cell_type="STEM", at_start=False)
        enterocyteness_regeneration_end_by_experiment[experiment.name.get_name()] \
            = _extract_probabilities_of_stem_cells(experiment, cell_type="ENTEROCYTE", at_start=False)

    # Collect ablation data
    stemness_ablation_by_experiment = dict()
    enterocyteness_ablation_by_experiment = dict()
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE_ABLATION):
        stemness_ablation_by_experiment[experiment.name.get_name()] \
            = _extract_probabilities_of_stem_cells(experiment, cell_type="STEM")
        enterocyteness_ablation_by_experiment[experiment.name.get_name()] \
            = _extract_probabilities_of_stem_cells(experiment, cell_type="ENTEROCYTE")

    # Append the data for each group (the function sum works, because ["a", "b"] + ["c"] = ["a", "b", "c"])
    stemness_ablation = sum(stemness_ablation_by_experiment.values(), [])
    stemness_regeneration_start = sum(stemness_regeneration_start_by_experiment.values(), [])
    stemness_regeneration_end = sum(stemness_regeneration_end_by_experiment.values(), [])
    enterocyteness_ablation = sum(enterocyteness_ablation_by_experiment.values(), [])
    enterocyteness_regeneration_start = sum(enterocyteness_regeneration_start_by_experiment.values(), [])
    enterocyteness_regeneration_end = sum(enterocyteness_regeneration_end_by_experiment.values(), [])

    figure = lib_figures.new_figure(size=(2.5, 4))
    ax_stemness, ax_enterocyteness = figure.subplots(nrows=2, ncols=1, sharex=True)
    ax_stemness.set_ylim(0.3, 0.7)
    violin = ax_stemness.violinplot([stemness_ablation, stemness_regeneration_start, stemness_regeneration_end],
                                    showmeans=False, showextrema=False, showmedians=False, widths=0.75)
    for body in violin['bodies']:
        body.set_facecolor(lib_figures.CELL_TYPE_PALETTE["STEM"])
        body.set_alpha(0.9)

    ax_stemness.set_xticks([1, 2, 3])
    _plot_organoid_averages(ax_stemness, stemness_ablation_by_experiment, 1)
    _plot_organoid_averages(ax_stemness, stemness_regeneration_start_by_experiment, 2)
    _plot_organoid_averages(ax_stemness, stemness_regeneration_end_by_experiment, 3)
    ax_stemness.set_ylabel("Stem cell likelihood")

    ax_enterocyteness.set_ylim(0, 0.6)
    violin = ax_enterocyteness.violinplot(
        [enterocyteness_ablation, enterocyteness_regeneration_start, enterocyteness_regeneration_end],
        showmeans=False, showextrema=False, showmedians=False, widths=0.75)
    for body in violin['bodies']:
        body.set_facecolor(lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"])
        body.set_alpha(0.9)

    ax_enterocyteness.set_xticks([1, 2, 3])
    _plot_organoid_averages(ax_enterocyteness, enterocyteness_ablation_by_experiment, 1)
    _plot_organoid_averages(ax_enterocyteness, enterocyteness_regeneration_start_by_experiment, 2)
    _plot_organoid_averages(ax_enterocyteness, enterocyteness_regeneration_end_by_experiment, 3)

    ax_enterocyteness.set_xticklabels(["Ablation start", "Regeneration start", "+24 h"])
    ax_enterocyteness.set_ylabel("Enterocyte likelihood")

    plt.show()


def _extract_probabilities_of_stem_cells(experiment: Experiment, *, cell_type: str, at_start: bool = True) -> List[
    float]:
    """Gets the probabilities for the given cell type of all predicted *STEM* cells."""
    cell_types = experiment.global_data.get_data("ct_probabilities")
    stemness = list()
    cell_type_index = cell_types.index(cell_type)
    stem_cell_index = cell_types.index("STEM")

    time_point = experiment.positions.first_time_point() if at_start else experiment.positions.last_time_point()
    for position in experiment.positions.of_time_point(time_point):

        ct_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
        if ct_probabilities is None:
            continue

        if numpy.argmax(ct_probabilities) == stem_cell_index:
            stemness.append(ct_probabilities[cell_type_index])

    return stemness


if __name__ == "__main__":
    main()
