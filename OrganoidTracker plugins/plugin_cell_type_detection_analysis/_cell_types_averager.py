"""Averages the probabilities over a number of time points, and used that to assign cell types."""
import numpy

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.util.moving_average import MovingAverage


_WINDOW_WIDTH_H = 2


def average_cell_types(window: Window):
    """Averages the cell type probabilities over several hours, to pick the most likely one."""
    for experiment in window.get_active_experiments():
        _average_experiment(experiment)
    window.get_gui_experiment().redraw_data()


def unaverage_cell_types(window: Window):
    """Removes the above averaging"""
    for experiment in window.get_active_experiments():
        _unaverage_experiment(experiment)
    window.get_gui_experiment().redraw_data()


def _average_experiment(experiment: Experiment):

    resolution = experiment.images.resolution()

    cell_type_names = experiment.global_data.get_data("ct_probabilities")
    if cell_type_names is None:
        raise UserError("No predicted cell types", f"The experiment {experiment.name} doesn't contain automatically"
                                                   f" predicted cell types")

    for track in experiment.links.find_all_tracks():
        # Create a list of lists. First list corresponds to the probability of cell_type_names[0] over time, etc.
        cell_type_probabilities = [list() for i in range(len(cell_type_names))]
        times_h = list()
        for position in track.positions():
            probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
            if probabilities is None:
                break

            times_h.append(position.time_point_number() * resolution.time_point_interval_h)
            for i, probability in enumerate(probabilities):
                cell_type_probabilities[i].append(probability)

        # Average them
        if len(cell_type_probabilities[0]) < 3:
            continue  # Too few probabilities
        averaged_cell_type_probabilities = [MovingAverage(times_h, probabilities, window_width=_WINDOW_WIDTH_H)
                                            for probabilities in cell_type_probabilities]

        # Now use to averages to assign cell types
        for position in track.positions():
            time_h = position.time_point_number() * resolution.time_point_interval_h
            probabilities = [averaged_probabilities.get_mean_at(time_h) for averaged_probabilities in averaged_cell_type_probabilities]
            if None in probabilities:
                continue  # Cannot do anything for this time point

            # Find most likely cell type in averaged probabilities
            cell_type = cell_type_names[numpy.argmax(probabilities)]
            position_markers.set_position_type(experiment.position_data, position, cell_type)


def _unaverage_experiment(experiment: Experiment):
    cell_type_names = experiment.global_data.get_data("ct_probabilities")
    if cell_type_names is None:
        raise UserError("No predicted cell types", f"The experiment {experiment.name} doesn't contain automatically"
                                                   f" predicted cell types")

    position_data = experiment.position_data
    for track in experiment.links.find_all_tracks():
        for position in track.positions():
            probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
            if probabilities is None:
                continue
            cell_type = cell_type_names[numpy.argmax(probabilities)]
            position_markers.set_position_type(position_data, position, cell_type)
