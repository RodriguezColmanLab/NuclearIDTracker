from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from typing import List, Optional, Union

import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.imaging import list_io
from organoid_tracker.linking_analysis.lineage_drawing import LineageDrawing

_DATA_FILE = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
#_DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"

_MAX_PLOTTED_TIME_POINT = 118  # Makes all experiments have the same length


def main():
    plt.rcParams['savefig.dpi'] = 180

    experiments = list_io.load_experiment_list_file(_DATA_FILE, load_images=False, max_time_point=_MAX_PLOTTED_TIME_POINT)
    experiment_count = list_io.count_experiments_in_list_file(_DATA_FILE)

    figure = lib_figures.new_figure()
    axes = figure.subplots(nrows=experiment_count, ncols=1)
    for ax, experiment in zip(axes, experiments):
        _draw_experiment(ax, experiment)
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


def _find_initial_enterocyteness(experiment: Experiment, track: LinkingTrack) -> float:
    cell_type_names = experiment.global_data.get_data("ct_probabilities")

    enterocyte_probabilities = list()
    for i, position in enumerate(track.positions()):
        probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
        if probabilities is None:
            continue
        enterocyte_probabilities.append(probabilities[cell_type_names.index("ENTEROCYTE")])
        if i > 5:
            break

    if len(enterocyte_probabilities) == 0:
        return 0
    return sum(enterocyte_probabilities) / len(enterocyte_probabilities)


def _last_time_point_number(starting_track: LinkingTrack) -> int:
    last_time_point_number = starting_track.last_time_point_number()
    for track in starting_track.find_all_descending_tracks(include_self=False):
        last_time_point_number = max(last_time_point_number, track.last_time_point_number())
    return last_time_point_number


def _draw_experiment(ax: Axes, experiment: Experiment):
    resolution = experiment.images.resolution()
    cell_type_names = experiment.global_data.get_data("ct_probabilities")

    # Override some colors, for maximal contrast
    lib_figures.CELL_TYPE_PALETTE["PANETH"] = "#d63031"

    first_time_point_number = experiment.positions.first_time_point_number()
    required_time_point_number = _MAX_PLOTTED_TIME_POINT // 2

    def filter_lineages(starting_track: LinkingTrack):
        return starting_track.first_time_point_number() == first_time_point_number and \
            _last_time_point_number(starting_track) >= required_time_point_number

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
    tracks = list(experiment.links.find_starting_tracks())
    tracks.sort(key=lambda track: _find_initial_enterocyteness(experiment, track), reverse=True)
    drawer = LineageDrawing(tracks)
    width = drawer.draw_lineages_colored(ax,
                                         color_getter=color_position,
                                         lineage_filter=filter_lineages,
                                         resolution=resolution,
                                         line_width=2)
    ax.set_xticks([])
    ax.set_ylabel("Time (h)")
    ax.set_ylim(y_max + 5, y_min)
    ax.set_xlim(-1, width + 1)
    #ax.set_facecolor("#dfe6e9")


if __name__ == "__main__":
    main()
