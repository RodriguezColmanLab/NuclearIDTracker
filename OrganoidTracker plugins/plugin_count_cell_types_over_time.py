import numpy
from matplotlib.figure import Figure

from organoid_tracker.core import UserError, max_none, min_none
from organoid_tracker.core.resolution import ImageTimings
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.plugin.registry import Registry


def get_menu_items(window: Window):
    return {
        "Graph//Cell types-Cell counts of each type over time...": lambda: _show_cell_counts_graph(window)
    }


class _CountsByTimePoint:
    _counts_by_cell_type: dict[str, numpy.ndarray]
    min_time_point_number: int
    max_time_point_number: int

    def __init__(self, min_time_point_number: int, max_time_point_number: int):
        self._counts_by_cell_type = dict()
        self.min_time_point_number = min_time_point_number
        self.max_time_point_number = max_time_point_number

    def _get_or_create_cell_type_array(self, cell_type: str) -> numpy.ndarray:
        if cell_type not in self._counts_by_cell_type:
            self._counts_by_cell_type[cell_type] = numpy.zeros(self.max_time_point_number - self.min_time_point_number + 1, dtype=int)
        return self._counts_by_cell_type[cell_type]

    def counts_over_time(self, cell_type: str) -> numpy.ndarray:
        """Returns the counts of the given cell type over time as a numpy array. Raises KeyError if the cell type
        is not found."""
        return self._counts_by_cell_type[cell_type]

    def count_cell(self, cell_type: str, time_point_number: int):
        if time_point_number < self.min_time_point_number or time_point_number > self.max_time_point_number:
            raise ValueError(f"Time point number {time_point_number} is out of range [{self.min_time_point_number}, {self.max_time_point_number}]")
        array = self._get_or_create_cell_type_array(cell_type)
        array[time_point_number - self.min_time_point_number] += 1

    def get_cell_types_by_count(self) -> list[str]:
        """Gets the cell types ordered by the total counts."""
        cell_types = list(self._counts_by_cell_type.keys())
        counts_by_cell_type = [self._counts_by_cell_type[cell_type].sum() for cell_type in cell_types]
        order = numpy.argsort(counts_by_cell_type)
        return [cell_types[i] for i in order]

    def time_point_numbers(self) -> numpy.ndarray:
        return numpy.arange(self.min_time_point_number, self.max_time_point_number + 1)


def _show_cell_counts_graph(window: Window):
    min_time_point_number = None
    max_time_point_number = None
    for experiment in window.get_active_experiments():
        min_time_point_number = min_none(experiment.positions.first_time_point_number(), min_time_point_number)
        max_time_point_number = max_none(experiment.positions.last_time_point_number(), max_time_point_number)
    if min_time_point_number is None or max_time_point_number is None:
        raise UserError("No positions found", "No positions were found. Did you forget to load the tracking data?")

    counts_by_time_point = _CountsByTimePoint(min_time_point_number, max_time_point_number)
    timings = None
    for experiment in window.get_active_experiments():
        if timings is None:
            timings = experiment.images.timings()
        for position, type_str in experiment.positions.find_all_positions_with_data("type"):
            counts_by_time_point.count_cell(type_str, position.time_point_number())

    dialog.popup_figure(window, lambda figure: _draw_figure(figure, window.registry, timings, counts_by_time_point),
                        size_cm=(9, 7))


def _draw_figure(figure: Figure, registry: Registry, timings: ImageTimings, counts_by_time_point: _CountsByTimePoint):
    cell_types_by_count = counts_by_time_point.get_cell_types_by_count()
    ax = figure.gca()

    bottom = None
    x_values = None
    legend_handles = []
    for cell_type in cell_types_by_count:
        marker = registry.get_marker_by_save_name(cell_type)
        color = marker.color.to_rgb_floats() if marker is not None else None
        display_name = marker.display_name if marker is not None else cell_type

        cell_type_counts = counts_by_time_point.counts_over_time(cell_type)
        if bottom is None:
            bottom = numpy.zeros_like(cell_type_counts)
            x_values = [timings.get_time_h_since_start(time_point_number) for time_point_number in counts_by_time_point.time_point_numbers()]

        patch = ax.fill_between(x_values, bottom, cell_type_counts + bottom, color=color, label=display_name, alpha=0.7, lw=0)
        legend_handles.insert(0, patch)
        bottom += cell_type_counts
    ax.legend(handles=legend_handles)
    ax.set_ylabel("Cell counts")
    ax.set_xlabel("Time (h)")

    ax.set_ylim(0, bottom.max() * 1.1)
    ax.set_xlim(x_values[0], x_values[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.tight_layout()