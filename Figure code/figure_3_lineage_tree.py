import math
from collections import defaultdict
from typing import Tuple, List, Optional, Union

from numpy import ndarray

import figure_lib
import matplotlib.cm
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.imaging import list_io
from organoid_tracker.linking_analysis.lineage_drawing import LineageDrawing
from organoid_tracker.position_analysis import intensity_calculator

_DATA_FILE = "../../Data/Predicted data.autlist"
_COLORMAP = matplotlib.cm.coolwarm
_AVERAGING_TIME_POINTS = 6
_EXPERIMENT_NAME = "x20190926pos01"


def _get_min_max_chance_per_cell_type(experiment: Experiment) -> Tuple[ndarray, ndarray]:
    probabilities = numpy.array([probabilities for position, probabilities in
                                 experiment.position_data.find_all_positions_with_data("ct_probabilities")])
    min_intensity = numpy.min(probabilities, axis=0)
    max_intensity = numpy.quantile(probabilities, q=0.95, axis=0)

    return min_intensity, max_intensity


def main():
    plt.rcParams['savefig.dpi'] = 180

    experiments = list_io.load_experiment_list_file(_DATA_FILE)

    figure = figure_lib.new_figure()
    ax = figure.gca()
    for experiment in experiments:
        if experiment.name.get_name() == _EXPERIMENT_NAME:
            _draw_experiment(ax, experiment)
            break
    plt.show()


def _search_probabilities(experiment: Experiment, position: Position) -> Optional[List[float]]:
    """Gives the cell type probabilities of the position. If not found, then it checks whether they are there in the
    previous or next time point, and returns those."""
    probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
    if probabilities is not None:
        return probabilities

    past_position = experiment.links.find_single_past(position)
    future_position = experiment.links.find_single_future(position)
    for i in range(10):
        if past_position is not None:
            probabilities = experiment.position_data.get_position_data(past_position, "ct_probabilities")
            if probabilities is not None:
                return probabilities
            past_position = experiment.links.find_single_past(past_position)

        if future_position is not None:
            probabilities = experiment.position_data.get_position_data(future_position, "ct_probabilities")
            if probabilities is not None:
                return probabilities
            future_position = experiment.links.find_single_future(future_position)

    return None


def _clip(value: Union[float, ndarray]) -> float:
    if value < 0:
        return float(0)
    if value > 1:
        return float(1)
    return float(value)


def _draw_experiment(ax: Axes, experiment: Experiment):
    resolution = experiment.images.resolution()
    cell_type_names = experiment.global_data.get_data("ct_probabilities")
    stem_index = cell_type_names.index("STEM")
    paneth_index = cell_type_names.index("PANETH")
    enterocyte_index = cell_type_names.index("ENTEROCYTE")
    min_probabilities, max_probabilities = _get_min_max_chance_per_cell_type(experiment)

    def filter_lineages(starting_track: LinkingTrack):
        min_time_point_number = starting_track.min_time_point_number()
        max_time_point_number = starting_track.max_time_point_number()
        for track in starting_track.find_all_descending_tracks():
            max_time_point_number = max(track.max_time_point_number(), max_time_point_number)
        return min_time_point_number == experiment.positions.first_time_point_number() \
            and max_time_point_number == experiment.positions.last_time_point_number()

    def color_position(time_point_number: int, track: LinkingTrack) -> MPLColor:
        position = track.find_position_at_time_point_number(time_point_number)
        cell_type_probabilities = _search_probabilities(experiment, position)
        if cell_type_probabilities is None:
            return 0.8, 0.8, 0.8

        stemness = (cell_type_probabilities[stem_index] - min_probabilities[stem_index]) /\
                   (max_probabilities[stem_index] - min_probabilities[stem_index])
        panethness = (cell_type_probabilities[paneth_index] - min_probabilities[paneth_index]) / \
                     (max_probabilities[paneth_index] - min_probabilities[paneth_index])
        enterocyteness = (cell_type_probabilities[enterocyte_index] - min_probabilities[enterocyte_index]) / \
                         (max_probabilities[enterocyte_index] - min_probabilities[enterocyte_index])
        return _clip(panethness), _clip(stemness), _clip(enterocyteness)

    y_min = 0
    y_max = experiment.positions.last_time_point_number() * resolution.time_point_interval_h

    ax.set_title(experiment.name.get_name())
    drawer = LineageDrawing(experiment.links)
    width = drawer.draw_lineages_colored(ax,
                                         color_getter=color_position,
                                         lineage_filter=filter_lineages,
                                         resolution=resolution,
                                         line_width=3)

    ax.set_xticks([])
    ax.set_ylabel("Time (h)")
    ax.set_ylim(y_max, y_min)
    ax.set_xlim(-1, width + 1)


if __name__ == "__main__":
    main()
