import os
from typing import Optional, List, Iterable

import numpy
import tifffile
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from numpy import ndarray

import lib_figures
import lib_data
from organoid_tracker.core import MPLColor
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.gui.location_map import LocationMap
from organoid_tracker.imaging import list_io
from organoid_tracker.linking_analysis.lineage_drawing import LineageDrawing

_INPUT_FILES = ["../../Data/Tracking data as controls/Dataset - full overweekend.autlist",
                "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"]
_OUTPUT_FOLDER = "../../Movies"

_REGENERATION_LINEAGES = {
    Position(346.00, 268.00, 3.00, time_point_number=0),
    Position(358.00, 290.00, 3.00, time_point_number=0),
    Position(386.00, 299.00, 3.00, time_point_number=0)
}
_CONTROL_LINEAGES = {
    Position(192.02, 299.39, 5, time_point_number=0)
}


def _search_probabilities(experiment: Experiment, position: Position) -> Optional[List[float]]:
    """Gives the cell type probabilities of the position. If not found, then it checks whether they are there in the
    previous or next time point, and returns those."""
    probabilities = experiment.positions.get_position_data(position, "ct_probabilities")
    if probabilities is not None:
        return probabilities

    past_position = experiment.links.find_single_past(position)
    future_position = experiment.links.find_single_future(position)
    for i in range(20):
        if past_position is not None:
            probabilities = experiment.positions.get_position_data(past_position, "ct_probabilities")
            if probabilities is not None:
                return probabilities
            past_position = experiment.links.find_single_past(past_position)

        if future_position is not None:
            probabilities = experiment.positions.get_position_data(future_position, "ct_probabilities")
            if probabilities is not None:
                return probabilities
            future_position = experiment.links.find_single_future(future_position)

    return None


def _has_any_position(experiment: Experiment, positions: Iterable[Position]) -> bool:
    for position in positions:
        if experiment.positions.contains_position(position):
            return True
    return False


def main():
    os.makedirs(_OUTPUT_FOLDER, exist_ok=True)
    for input_file in _INPUT_FILES:
        for experiment in list_io.load_experiment_list_file(input_file):
            if not _has_any_position(experiment, _CONTROL_LINEAGES | _REGENERATION_LINEAGES):
                continue

            print(f"\nWorking on {experiment.name}...")
            cell_type_names = experiment.global_data.get_data("ct_probabilities")

            movie_array = None
            figure = plt.figure(figsize=(2.4, 2), dpi=150, facecolor="black")
            ax_lineages = figure.gca()

            def lineage_filter(track: LinkingTrack) -> bool:
                # Only take lineages that span the entire time lapse
                # min_time_point_number = experiment.positions.first_time_point_number()
                # max_time_point_number = experiment.positions.last_time_point_number()
                #
                # track_length = track.max_time_point_number() - track.min_time_point_number()
                # return track.will_divide() or track_length > max_time_point_number / 2

                return track.find_first_position() in _REGENERATION_LINEAGES or track.find_first_position() in _CONTROL_LINEAGES

            def label_getter(track: LinkingTrack):
                if len(track.get_previous_tracks()) > 0:
                    return None
                p = track.find_first_position()
                return f"{p.x:.2f}, {p.y:.2f}, {p.z:.2f}, {p.time_point_number()}".replace(".00", "")

            def color_position(time_point_number: int, track: LinkingTrack) -> MPLColor:
                # If the time point is the first or last, then we don't want to use the probabilities of that time point
                # They might be unreliable due to the cell dividing or dying
                if track.will_divide() and time_point_number + 1 >= track.last_time_point_number():
                    return 0.8, 0.8, 0.8  # Always draw divisions in gray

                position = track.find_position_at_time_point_number(time_point_number)
                cell_type_probabilities = _search_probabilities(experiment, position)
                if cell_type_probabilities is None:
                    return 0.8, 0.8, 0.8

                # stem_to_ec = lib_data.find_stem_to_ec_location(cell_type_names, cell_type_probabilities)
                # if stem_to_ec is None:
                #     return 0.8, 0.8, 0.8
                # return lib_figures.get_stem_to_ec_color(stem_to_ec)
                return lib_figures.get_mixed_cell_type_color(cell_type_names, cell_type_probabilities)

            time_points = list(experiment.positions.time_points())
            for i, time_point in enumerate(time_points):
                ax_lineages.clear()
                ax_lineages.set_facecolor("black")

                print(time_point.time_point_number(), end="  ")
                cutoff_h = experiment.images.timings().get_time_h_since_start(time_point)

                location_map = LocationMap()
                width = LineageDrawing(experiment.links).draw_lineages_colored(ax_lineages,
                                                                               timings=experiment.images.timings(),
                                                                               lineage_filter=lineage_filter,
                                                                               #label_getter=label_getter,
                                                                               color_getter=color_position,
                                                                               location_map=location_map,
                                                                               line_width=2)

                # Redraw lineage trees with time cutoff
                _redraw_lineages(ax_lineages, cutoff_h)

                # Set axis limits
                max_y = experiment.images.timings().get_time_h_since_start(experiment.last_time_point_number())
                ax_lineages.set_xlim(-2, width)
                ax_lineages.set_ylim(max_y, 0)

                # Decoration
                ax_lineages.set_xticks([])
                ax_lineages.set_ylabel("Time (h)", color="#dfe6e9")
                ax_lineages.spines[['bottom', 'right', 'top']].set_visible(False)
                ax_lineages.spines['left'].set_color("#dfe6e9")
                ax_lineages.tick_params(color="#dfe6e9", labelcolor="#dfe6e9")

                figure.tight_layout()
                array = _to_bitmap(figure)
                if movie_array is None:
                    movie_array = numpy.zeros_like(array,
                                                   shape=(len(time_points), array.shape[0], array.shape[1], array.shape[2]))
                movie_array[i] = array

            tifffile.imwrite(os.path.join(_OUTPUT_FOLDER, "Lineage trees " + experiment.name.get_save_name() + ".tif"), movie_array)


def _redraw_lineages(ax: Axes, cutoff_h: float):
    """Clears the axes, and then draws the lineages again up to cutoff_h."""
    new_line_collections = list()
    for child in ax.get_children():
        if isinstance(child, LineCollection):
            old_colors = child.get_color()
            new_lines = list()
            new_colors = list()
            for color, ((x_from, y_from), (x_to, y_to)) in zip(old_colors, child.get_segments()):
                if y_from < cutoff_h:
                    new_lines.append(((x_from, y_from), (x_to, min(cutoff_h, y_to))))
                    new_colors.append(color)
            new_line_collections.append(LineCollection(new_lines, colors=new_colors, linewidths=child.get_linewidth()))
    ax.clear()
    for new_line_collection in new_line_collections:
        ax.add_collection(new_line_collection)


def _to_bitmap(figure: Figure) -> ndarray:
    figure.tight_layout()
    figure.canvas.draw()

    # Now we can save it to a numpy array.
    buf = numpy.frombuffer(figure.canvas.buffer_rgba(), dtype=numpy.uint8)
    width, height = figure.canvas.get_width_height()
    data = buf.reshape((height, width, 4))[:, :, :3]

    return data


if __name__ == "__main__":
    main()
