from collections import defaultdict
from typing import List, Optional

import matplotlib.colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
from organoid_tracker.core import Color
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.util.moving_average import MovingAverage

_AVERAGING_WINDOW_WIDTH_H = 5
_DATA_FILE = "../../Data/Predicted data.autlist"
_EXPERIMENT_NAME = "x20190926pos01"
_PLOTTED_POSITIONS = [Position(89.73, 331.37, 5.00, time_point_number=331),
                      Position(125.56, 294.27, 12.00, time_point_number=331),
                      Position(234.63, 343.08, 8.00, time_point_number=331)]

def main():
    experiments = list_io.load_experiment_list_file(_DATA_FILE)

    figure = lib_figures.new_figure()
    axes = figure.subplots(nrows=len(_PLOTTED_POSITIONS), ncols=1, sharey=True, sharex=True)
    for experiment in experiments:
        if experiment.name.get_name() == _EXPERIMENT_NAME:
            for ax, position in zip(axes, _PLOTTED_POSITIONS):
                _draw_position(ax, experiment, position)
            break
    axes[0].legend()
    axes[-1].set_xlabel("Time (h)")
    axes[len(axes) // 2].set_ylabel("Probability")
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


def _draw_position(ax: Axes, experiment: Experiment, last_position: Position):
    timings = experiment.images.timings()
    cell_type_names = experiment.global_data.get_data("ct_probabilities")

    t_values = list()
    probabilities = defaultdict(list)
    for position in experiment.links.iterate_to_past(last_position):
        cell_type_probabilities = _search_probabilities(experiment, position)
        if cell_type_probabilities is None:
            continue

        if len(experiment.links.find_futures(position)) > 1:
            # Found a cell division, draw a vertical line
            ax.axvline(timings.get_time_h_since_start(position.time_point_number()), color="gray", linestyle="--")

        t_values.append(timings.get_time_h_since_start(position.time_point_number()))
        for i, cell_type_name in enumerate(cell_type_names):
            probabilities[cell_type_name].append(cell_type_probabilities[i])

    for cell_type_name in cell_type_names:
        MovingAverage(t_values, probabilities[cell_type_name],
                      window_width=_AVERAGING_WINDOW_WIDTH_H).plot(ax,
                                                                   color=_parse_color(
                                                                       lib_figures.CELL_TYPE_PALETTE[cell_type_name]),
                                                                   label=lib_figures.style_cell_type_name(cell_type_name))
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.33, 0.66, 1])


def _parse_color(color: str) -> Color:
    r, g, b = matplotlib.colors.to_rgb(color)
    return Color(int(r * 255), int(g * 255), int(b * 255))


if __name__ == "__main__":
    main()
