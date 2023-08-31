import os

import numpy

import lib_data
import lib_models
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io, io
from organoid_tracker.position_analysis import position_markers

_MODEL_FOLDER = r"../Data/Models/epochs-1-neurons-0"
_INPUT_FILE = "../Data/Testing data - input - CD24 and Lgr5.autlist"
_OUTPUT_FOLDER = "../Data/Testing data - output - CD24 and Lgr5"

# For filtering out missegmentations
_MIN_VOLUME_UM3 = 40
_MAX_VOLUME_UM3 = 1000


def predict_organoid(experiment: Experiment):
    # Delete existing cell types
    experiment.position_data.delete_data_with_name("type")

    # Load the model
    model = lib_models.load_model(_MODEL_FOLDER)
    input_names = model.get_input_output().input_mapping
    volume_index = input_names.index("volume_um3")

    # Construct data arrays in the desired format
    data_arrays = numpy.zeros((len(experiment.positions), len(input_names)))
    positions_corresponding_to_data_array = list()
    i = 0
    for position in experiment.positions:
        data_array = lib_data.get_data_array(experiment.position_data, position, input_names)
        if data_array is None:
            continue
        volume_um3 = data_array[volume_index]
        if volume_um3 < _MIN_VOLUME_UM3 or volume_um3 > _MAX_VOLUME_UM3:
            continue  # Segmentation fail

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
        experiment.position_data.set_position_data(position, "ct_probabilities",
                                                   [float(prob) for prob in probability_by_cell_type])
        cell_type = cell_types[numpy.argmax(probability_by_cell_type)]
        position_markers.set_position_type(experiment.position_data, position, cell_type)


def main():
    os.makedirs(_OUTPUT_FOLDER, exist_ok=True)
    for experiment in list_io.load_experiment_list_file(_INPUT_FILE):
        print(f"Working on {experiment.name}...")
        predict_organoid(experiment)
        io.save_data_to_json(experiment,
                             os.path.join(_OUTPUT_FOLDER, f"{experiment.name.get_save_name()}.{io.FILE_EXTENSION}"))


if __name__ == "__main__":
    main()
