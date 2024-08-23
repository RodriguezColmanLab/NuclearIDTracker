import math
from typing import List, Optional, Union

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.gui.location_map import LocationMap
from organoid_tracker.imaging import list_io
from organoid_tracker.linking_analysis.lineage_drawing import LineageDrawing

_DATA_FILE = "../../Data/Predicted data.autlist"
_EXPERIMENT_NAME = "x20190926pos01"
_PLOTTED_LINEAGE_TREES = [Position(192.02, 299.39, 5, time_point_number=1),  # Both absorptive and secretory cells
                          Position(145.97, 333.90, 16, time_point_number=1),
                          Position(203.45, 349.31, 12, time_point_number=1)]

# Will have a small number at the bottom of the lineage tree
_ANNOTATED_POSITIONS = [Position(89.73, 331.37, 5.00, time_point_number=331),
                        Position(125.56, 294.27, 12.00, time_point_number=331),
                        Position(234.63, 343.08, 8.00, time_point_number=331)]


def main():
    plt.rcParams['savefig.dpi'] = 180

    experiments = list_io.load_experiment_list_file(_DATA_FILE)

    figure = lib_figures.new_figure()
    ax = figure.gca()
    for experiment in experiments:
        if experiment.name.get_name() == _EXPERIMENT_NAME:
            _draw_experiment(ax, experiment)
            break
    plt.show()


def _scale_probabilities(probabilities: List[float]) -> List[float]:
    # Scales the probabilities so that the max is 1, and everything less than 50% of the max is 0
    # In this way, we mostly see the dominant cell type

    max_probability = max(probabilities)
    min_plotted_probability = max_probability * 0.5

    probabilities = [(probability - min_plotted_probability) / (max_probability - min_plotted_probability) for probability in probabilities]

    return [_clip(probability) for probability in probabilities]


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

    # Override some colors, for maximal contrast
    lib_figures.CELL_TYPE_PALETTE["PANETH"] = "#d63031"

    def filter_lineages(starting_track: LinkingTrack):
        return starting_track.find_first_position() in _PLOTTED_LINEAGE_TREES

    def color_position(time_point_number: int, track: LinkingTrack) -> MPLColor:
        # If the time point is the first or last, then we don't want to use the probabilities of that time point
        # They might be unreliable due to the cell dividing or dying
        if track.will_divide() and time_point_number + 1 >= track.last_time_point_number():
            return lib_figures.CELL_TYPE_PALETTE["STEM"]  # Always draw divisions in the color of stem cells

        position = track.find_position_at_time_point_number(time_point_number)
        cell_type_probabilities = _search_probabilities(experiment, position)
        if cell_type_probabilities is None:
            return 0.8, 0.8, 0.8

        return lib_figures.get_mixed_cell_type_color(cell_type_names, cell_type_probabilities)

    y_min = 0
    y_max = experiment.positions.last_time_point_number() * resolution.time_point_interval_h

    ax.set_title(experiment.name.get_name())
    drawer = LineageDrawing(experiment.links)
    location_map = LocationMap()
    width = drawer.draw_lineages_colored(ax,
                                         color_getter=color_position,
                                         lineage_filter=filter_lineages,
                                         resolution=resolution,
                                         location_map=location_map,
                                         line_width=3)
    for i, position in enumerate(_ANNOTATED_POSITIONS):
        x, y = location_map.find_object(position)
        if x is not None:
            ax.text(x, y_max + 1, str(i + 1), horizontalalignment="center", verticalalignment="top")
    ax.set_xticks([])
    ax.set_ylabel("Time (h)")
    ax.set_ylim(y_max + 5, y_min)
    ax.set_xlim(-1, width + 1)


if __name__ == "__main__":
    main()
