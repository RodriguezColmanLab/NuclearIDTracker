from collections import defaultdict
from typing import Dict, List, Optional

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from organoid_tracker.core import UserError, Color
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.util.moving_average import MovingAverage
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer

_WINDOW_WIDTH_H = 4


class _ProbabilitiesOverTime:
    """Collects the predicted probabilities of all cell types."""
    probabilities_by_cell_type: Dict[str, List[float]]
    time_hours: List[float]
    parent: Optional["_ProbabilitiesOverTime"] = None

    def __init__(self, experiment: Experiment, track: LinkingTrack):
        # Collect the available cell types
        cell_type_names = experiment.global_data.get_data("ct_probabilities")
        if cell_type_names is None:
            raise UserError("No predicted cell types", f"The experiment {experiment.name} doesn't contain automatically"
                                                       f" predicted cell types")

        # Collect the probabilities
        position_data = experiment.position_data
        resolution = experiment.images.resolution()
        self.probabilities_by_cell_type = defaultdict(list)
        self.time_hours = list()
        for position in track.positions():
            cell_type_probabilities = position_data.get_position_data(position, "ct_probabilities")
            if cell_type_probabilities is None:
                continue

            for cell_type, probability in zip(cell_type_names, cell_type_probabilities):
                self.probabilities_by_cell_type[cell_type].append(probability)
            self.time_hours.append(position.time_point_number() * resolution.time_point_interval_h)

        # Run for parent
        parent_tracks = track.get_previous_tracks()
        if len(parent_tracks) == 1:
            self.parent = _ProbabilitiesOverTime(experiment, parent_tracks.pop())

    def n_most_likely_cell_types(self, *, n: int) -> Dict[str, List[float]]:
        """Gets the probabiliites for the N most likely cell types."""
        sorted_values = sorted(self.probabilities_by_cell_type.items(), key=lambda item: sum(item[1]), reverse=True)
        if len(sorted_values) > n:
            sorted_values = sorted_values[0:n]
        return dict(sorted_values)


class CellTypeVisualizer(ExitableImageVisualizer):
    """Used to find how the cell type changes over time. Double-click on a cell to view its cell type probabilities."""

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            super()._on_mouse_click(event)
            return

        position = self._get_position_at(event.xdata, event.ydata)
        if position is None:
            self.update_status("No cell found here.")
            return  # No cell selected

        self._popup_cell_graph(position)

    def _popup_cell_graph(self, position: Position):
        track = self._experiment.links.get_track(position)
        if track is None:
            self.update_status("No links found for this position.")
            return

        all_probabilities = _ProbabilitiesOverTime(self._experiment, track)

        gui_experiment = self.get_window().get_gui_experiment()
        dialog.popup_figure(gui_experiment, lambda figure: self._plot(figure, all_probabilities), size_cm=(20, 7))


    def _plot(self, figure: Figure, all_probabilities: _ProbabilitiesOverTime):
        ax: Axes = figure.gca()

        show_labels = True
        while all_probabilities is not None:
            self._plot_probabilities(ax, all_probabilities, show_labels)
            all_probabilities = all_probabilities.parent
            show_labels = False  # Add cell type labels only for the first one
            if all_probabilities is not None:
                # Plot line at division
                ax.axvline(x=max(all_probabilities.time_hours), color="black", linewidth=3)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Probability")
        ax.legend()





    def _plot_probabilities(self, ax: Axes, probabiliities: _ProbabilitiesOverTime, show_label: bool = False):
        for cell_type, probabilities in probabiliities.n_most_likely_cell_types(n=4).items():
            color = self._get_color_for_cell_type(cell_type)
            label = cell_type.lower().replace("_", " ") if show_label else None
            MovingAverage(probabiliities.time_hours, probabilities, window_width=_WINDOW_WIDTH_H,
                          x_step_size=0.1) \
                .plot(ax, color=color, label=label)

    def _get_color_for_cell_type(self, cell_type: str) -> Color:
        color = Color.black()
        marker = self.get_window().registry.get_marker_by_save_name(cell_type.upper())
        if marker is not None and marker.applies_to(Position):
            color = marker.color
        return color
