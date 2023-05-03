import math
import os
from typing import Optional, List

import numpy
from numpy import ndarray

import lib_models
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.imaging import list_io, io
from organoid_tracker.position_analysis import position_markers

_MODEL_FOLDER = r"../Data/Models/epochs-1-neurons-0"
_INPUT_FILE = "../Data/Training data.autlist"
_OUTPUT_FOLDER = "../Data/Cell tracks with predicted types"


def predict_organoid(experiment: Experiment):
    # Delete existing cell types
    experiment.position_data.delete_data_with_name("type")

    # Load the model
    model = lib_models.load_model(_MODEL_FOLDER)
    input_names = model.get_input_output().input_mapping

    # Construct data arrays in the desired format
    data_arrays = numpy.zeros((len(experiment.positions), len(input_names)))
    positions_corresponding_to_data_array = list()
    i = 0
    for position in experiment.positions:
        data_array = _get_data_array(experiment.position_data, position, input_names)
        if data_array is not None:
            data_arrays[i] = data_array
            positions_corresponding_to_data_array.append(position)
            i += 1
    data_arrays = data_arrays[0:i]

    # Apply the model
    probabilities = model.predict(data_arrays)
    cell_types = model.get_input_output().cell_type_mapping

    # Store the predicted cell types
    experiment.global_data.set_data("ct_probabilities", cell_types)
    for position, probability_by_cell_type in zip(positions_corresponding_to_data_array, probabilities):
        experiment.position_data.set_position_data(position, "ct_probabilities", list(probability_by_cell_type))
        cell_type = cell_types[numpy.argmax(probability_by_cell_type)]
        position_markers.set_position_type(experiment.position_data, position, cell_type)


def _get_data_array(position_data: PositionData, position: Position, input_names: List[str]) -> Optional[ndarray]:
    array = numpy.empty(len(input_names), dtype=numpy.float32)
    for i, name in enumerate(input_names):
        value = None
        if name == "sphericity":
            # Special case, we need to calculate
            volume = position_data.get_position_data(position, "volume_um3")
            surface = position_data.get_position_data(position, "surface_um2")
            if volume is not None and surface is not None:
                value = math.pi ** (1/3) * (6 * volume) ** (2/3) / surface
        else:
            # Otherwise, just look up
            value = position_data.get_position_data(position, name)

        if value is None:
            return None  # Abort, a value is missing

        if name in {"neighbor_distance_variation", "solidity", "sphericity", "ellipticity", "intensity_factor"}\
                or name.endswith("_local"):
            # Ratios should be an exponential, as the analysis will log-transform the data
            value = math.exp(value)

        array[i] = value
    return array


def main():
    os.makedirs(_OUTPUT_FOLDER, exist_ok=True)
    for experiment in list_io.load_experiment_list_file(_INPUT_FILE):
        print(f"Working on {experiment.name}...")
        predict_organoid(experiment)
        io.save_data_to_json(experiment, os.path.join(_OUTPUT_FOLDER, f"{experiment.name.get_save_name()}.{io.FILE_EXTENSION}"))


if __name__ == "__main__":
    main()
