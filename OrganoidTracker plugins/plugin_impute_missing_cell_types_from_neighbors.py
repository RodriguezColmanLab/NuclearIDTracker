"""Calculates the cell type transitin matrix for the cell types stem, enterocyte, Paneth, other secretory."""
import json
from collections import defaultdict
from typing import Dict, Tuple, Optional, Union

import numpy

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking import nearby_position_finder
from organoid_tracker.position_analysis import position_markers


def get_menu_items(window: Window):
    return {
        "Tools//Process-Cell types//Transitions-Impute missing cell types from neighbors...": lambda: _impute_missing_cell_types(window)
    }


def _impute_probabilities(experiment: Experiment, position: Position, distance_um):
    positions_at_same_time_point = experiment.positions.of_time_point(position.time_point())
    resolution = experiment.images.resolution()
    neighboring_cells = nearby_position_finder.find_closest_n_positions(positions_at_same_time_point, around=position,
                                                                        max_amount=50, resolution=resolution,
                                                                        max_distance_um=distance_um)

    summed_probabilities = None
    count = 0
    for neighbor in neighboring_cells:
        neighbor_probabilities = experiment.position_data.get_position_data(neighbor, "ct_probabilities")
        is_imputed = experiment.position_data.get_position_data(neighbor, "ct_probabilities_imputed")

        if neighbor_probabilities is None or is_imputed:
            continue

        if summed_probabilities is None:
            summed_probabilities = numpy.array(neighbor_probabilities)
        else:
            summed_probabilities += neighbor_probabilities
        count += 1

    if count > 0:
        # Set the average probabilities
        summed_probabilities /= count
        experiment.position_data.set_position_data(position, "ct_probabilities", summed_probabilities)
        experiment.position_data.set_position_data(position, "ct_probabilities_imputed", True)

        # Set the cell type
        cell_types = experiment.global_data.get_data("ct_probabilities")
        cell_type = cell_types[numpy.argmax(summed_probabilities)]
        position_markers.set_position_type(experiment.position_data, position, cell_type)


def _impute_missing_cell_types(window: Window):
    distance_um = dialog.prompt_float("Neighbor distance", "For any missing cell type, we will take the"
                                      " average probabilities of the cells in the neighborhood. Up to which distance"
                                      " (in micrometers) should we impute missing cell types?")

    for experiment in window.get_active_experiments():
        cell_types = experiment.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise UserError("No cell types",
                            "No cell type probabilities available. Please run the cell type classifier first.")

        for time_point in experiment.time_points():
            for position in experiment.positions.of_time_point(time_point):
                probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
                is_imputed = experiment.position_data.get_position_data(position, "ct_probabilities_imputed")
                if probabilities is None or is_imputed:
                    _impute_probabilities(experiment, position, distance_um)
