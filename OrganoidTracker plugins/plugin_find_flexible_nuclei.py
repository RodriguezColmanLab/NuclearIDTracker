import matplotlib
import numpy
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from typing import Dict, Callable, Optional, Any, List

from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer

_MEASUREMENT_TIME_POINTS = 150


def get_menu_items(window: Window) -> Dict[str, Callable]:
    return {
        "View//Analyze-Flexible nuclei...": lambda: _show_nucleus_flexibility(window),
    }


def _show_nucleus_flexibility(window: Window):
    activate(_NucleusFlexibilityAnalyzer(window))


def _calculate_flexibility(length_values: List[float]) -> float:
    # Return the coefficient of variation
    if len(length_values) == 1:
        return 0
    return numpy.std(length_values, ddof=1) / numpy.mean(length_values)


def _none_or_zero(value: Optional[float]) -> bool:
    return value is None or value == 0


class _NucleusFlexibilityAnalyzer(ExitableImageVisualizer):
    """For plotting how flexible nuclei are. Select a position to get started."""

    _selected_position: Optional[Position] = None
    _track_to_flexibility: Dict[Position, float]

    def __init__(self, window: Window):
        super().__init__(window)
        self._track_to_flexibility = dict()

    def _calculate_time_point_metadata(self):
        self._track_to_flexibility.clear()
        for position in self._experiment.positions.of_time_point(self._time_point):
            flexibility = self._get_nucleus_flexibility(position)
            if flexibility is not None:
                self._track_to_flexibility[position] = flexibility

    def _get_nucleus_flexibility(self, starting_position: Position) -> Optional[float]:
        position_data = self._experiment.position_data
        major_axis_lengths = list()
        minor_axis_lengths = list()
        intermediate_axis_lengths = list()
        for position in self._experiment.links.iterate_to_past(starting_position):
            major_axis_length = position_data.get_position_data(position, "major_axis_length_um")
            minor_axis_length = position_data.get_position_data(position, "minor_axis_length_um")
            intermediate_axis_length = position_data.get_position_data(position, "intermediate_axis_length_um")
            if not _none_or_zero(major_axis_length) and not _none_or_zero(minor_axis_length) and not _none_or_zero(intermediate_axis_length):
                major_axis_lengths.append(major_axis_length)
                minor_axis_lengths.append(minor_axis_length)
                intermediate_axis_lengths.append(intermediate_axis_length)
            if starting_position.time_point_number() - position.time_point_number() > _MEASUREMENT_TIME_POINTS:
                break

        if len(major_axis_lengths) < _MEASUREMENT_TIME_POINTS * 0.9:
            return None

        major_axis_flexibility = _calculate_flexibility(major_axis_lengths)
        minor_axis_flexibility = _calculate_flexibility(minor_axis_lengths)
        intermediate_axis_flexibility = _calculate_flexibility(intermediate_axis_lengths)
        return (major_axis_flexibility + minor_axis_flexibility + intermediate_axis_flexibility) / 3

    def _on_mouse_single_click(self, event: MouseEvent):
        if event.button == 1:
            self._selected_position = self._get_position_at(event.xdata, event.ydata)
            self.draw_view()
        else:
            super()._on_mouse_single_click(event)

    def _draw_extra(self):
        if self._selected_position is not None:
            self._draw_selection(self._selected_position, color='lime')

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if self._selected_position is not None:
            return True
        flexibility = self._track_to_flexibility.get(position)
        if flexibility is not None:
            color = matplotlib.cm.jet(flexibility * 10)
            self._draw_selection(position, color)
        return True

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Graph//Lines-Plot axis lengths": self._plot_axis_lengths,
            "Graph//Lines-Plot all parameters": self._plot_all_parameters
        }

    def _plot_axis_lengths(self):
        if self._selected_position is None:
            self.update_status('No position selected.')
            return
        track = self._experiment.links.get_track(self._selected_position)
        if track is None:
            self.update_status('Selected position is not part of a track. Please select a different position.')
            return

        position_data = self._experiment.position_data
        time_point_numbers = list()
        major_axis_lengths = list()
        minor_axis_lengths = list()
        intermediate_axis_lengths = list()
        division_time_points = list()
        while True:
            # Collect data for current track
            for position in reversed(list(track.positions())):
                major_axis_length = position_data.get_position_data(position, "major_axis_length_um")
                minor_axis_length = position_data.get_position_data(position, "minor_axis_length_um")
                intermediate_axis_length = position_data.get_position_data(position, "intermediate_axis_length_um")
                if major_axis_length is not None and minor_axis_length is not None and intermediate_axis_length is not None:
                    time_point_numbers.append(position.time_point_number())
                    major_axis_lengths.append(major_axis_length)
                    minor_axis_lengths.append(minor_axis_length)
                    intermediate_axis_lengths.append(intermediate_axis_length)

            # Switch to previous track
            previous_tracks = track.get_previous_tracks()
            if len(previous_tracks) > 0:
                division_time_points.append(track.first_time_point_number())
                track = previous_tracks.pop()
            else:
                break

        if len(time_point_numbers) == 0:
            self.update_status('No data for selected parameter in this track.')
            return

        def draw_function(figure: Figure):
            max_value = max(max(major_axis_lengths), max(minor_axis_lengths), max(intermediate_axis_lengths))

            ax = figure.gca()
            ax.plot(time_point_numbers, major_axis_lengths, label='Major axis length', linewidth=2)
            ax.plot(time_point_numbers, intermediate_axis_lengths, label='Intermediate axis length', linewidth=2)
            ax.plot(time_point_numbers, minor_axis_lengths, label='Minor axis length', linewidth=2)
            ax.legend()
            ax.set_ylim(0, max_value * 1.1)
            ax.set_xlim(self._experiment.first_time_point_number(), self._experiment.last_time_point_number())
            ax.set_xlabel('Time point')
            ax.set_ylabel('Length (μm)')
            for division_time_point in division_time_points:
                ax.axvline(division_time_point, color='gray', linestyle='--')

        dialog.popup_figure(self._window, draw_function)

    def _plot_all_parameters(self):
        if self._selected_position is None:
            self.update_status('No position selected.')
            return
        track = self._experiment.links.get_track(self._selected_position)
        if track is None:
            self.update_status('Selected position is not part of a track. Please select a different position.')
            return

        # Collect parameter names
        position_data = self._experiment.position_data
        parameter_names = list()
        for parameter_name, parameter_type in position_data.get_data_names_and_types().items():
            if parameter_name.endswith('_local') or "penalty" in parameter_name or "probability" in parameter_name or "error" in parameter_name:
                # Ignore OrganoidTracker internal parameters and local parameters
                continue
            if parameter_name == "organoid_relative_z_um":
                # Ignore z position - not used in this analysis
                continue
            if issubclass(parameter_type, float):
                parameter_names.append(parameter_name)

        # Collect parameter values
        time_point_numbers = list()
        all_parameter_values = list()
        division_time_points = list()
        while True:
            # Collect data for current track
            for position in reversed(list(track.positions())):
                parameter_values = [None] * len(parameter_names)
                for i, parameter_name in enumerate(parameter_names):
                    parameter_values[i] = position_data.get_position_data(position, parameter_name)
                if None not in parameter_values:
                    time_point_numbers.append(position.time_point_number())
                    all_parameter_values.append(parameter_values)
                else:
                    print(parameter_names, parameter_values)

            # Switch to previous track
            previous_tracks = track.get_previous_tracks()
            if len(previous_tracks) > 0:
                division_time_points.append(track.first_time_point_number())
                track = previous_tracks.pop()
            else:
                break

        if len(time_point_numbers) == 0:
            self.update_status('No data for selected parameter in this track.')
            return

        def draw_function(figure: Figure):

            axes = figure.subplots(nrows=len(parameter_names), ncols=2, sharex=True, squeeze=False)
            for i, parameter_name in enumerate(parameter_names):
                ax_left, ax_right = axes[i]
                parameter_values = [values[i] for values in all_parameter_values]
                ax_left.plot(time_point_numbers, parameter_values, linewidth=2)
                ax_left.set_ylim(0, max(parameter_values) * 1.1)
                ax_left.set_xlim(self._experiment.first_time_point_number(), self._experiment.last_time_point_number())
                ax_left.set_ylabel(parameter_name, rotation="horizontal", ha='right')

                parameter_delta_values = numpy.diff(parameter_values)
                ax_right.plot(time_point_numbers[1:], parameter_delta_values, linewidth=1, color="black")
                quantile_cutoff = max(abs(numpy.quantile(parameter_delta_values, 0.99)), abs(numpy.quantile(parameter_delta_values, 0.01)))
                ax_right.set_ylim(-quantile_cutoff * 5, quantile_cutoff * 5)

                for division_time_point in division_time_points:
                    ax_left.axvline(division_time_point, color='gray', linestyle='--')
                    ax_right.axvline(division_time_point, color='gray', linestyle='--')


            axes[-1][0].set_xlim(self._experiment.first_time_point_number(), self._experiment.last_time_point_number())
            axes[-1][0].set_xlabel('Time point')

            figure.tight_layout()
            figure.subplots_adjust(hspace=0)

        dialog.popup_figure(self._window, draw_function, size_cm = (25, 25))
