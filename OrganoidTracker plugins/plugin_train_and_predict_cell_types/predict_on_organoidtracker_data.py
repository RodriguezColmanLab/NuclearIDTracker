import json
import os
from typing import NamedTuple

import numpy

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window
from organoid_tracker.util.run_script_creator import create_run_script
from . import lib_data
from . import lib_models
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io, io
from organoid_tracker.position_analysis import position_markers


class _ParsedConfig(NamedTuple):
    model_folder: str
    output_folder: str
    tracking_input_file: str


def _predict_organoid(experiment: Experiment, model: lib_models.OurModel):
    # Delete existing cell types
    experiment.position_data.delete_data_with_name("type")

    # Get input parameter names
    input_names = model.get_input_output().input_mapping

    # Construct data arrays in the desired format
    data_arrays = numpy.zeros((len(experiment.positions), len(input_names)))
    positions_corresponding_to_data_array = list()
    i = 0
    for position in experiment.positions:
        data_array = lib_data.get_data_array(experiment.position_data, position, input_names)
        if data_array is None:
            continue

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


def _parse_config(config: ConfigFile) -> _ParsedConfig:
    """Parses the config, and adds any missing comments or default settings."""
    model_folder = config.get_or_default("types_model", "nuclei",
                                         comment="Path to the model used to predict cell types.")
    dataset_path = config.get_or_default("dataset_to_segment", "Dataset" + list_io.FILES_LIST_EXTENSION,
                                         comment="Path to the dataset that you are using.")
    output_folder = config.get_or_default("output_folder", "Predictions/",
                                          comment="Path to the folder that will contain the tracking files with the"
                                                  " predictions.")
    return _ParsedConfig(
        model_folder=model_folder,
        tracking_input_file=dataset_path,
        output_folder=output_folder
    )


def create_prediction_script(window: Window):
    """Creates a segmentation script in the GUI. Called from get_menu_items()."""
    for experiment in window.get_active_experiments():
        if not "ellipticity" in experiment.position_data.get_data_names_and_types():
            raise UserError("Missing extracted parameters", f"You haven't extracted the shape parameters yet from the"
                                                            f" CellPose segmentation for the experiment"
                                                            f" \"{experiment.name}\". Please do so first.")

    # Find model directory
    if not dialog.prompt_confirmation("Cell type prediction", "We will first ask you where you have stored the cell"
                                                              " type prediction model.\n\nPlease select the appropriate"
                                                              " folder."):
        return
    types_model_path = dialog.prompt_directory("Cell type prediction model")
    if types_model_path is None:
        return
    lib_models.load_model(types_model_path)  # Load the model as a test

    # Creates an output folder
    if not dialog.prompt_confirmation("Cell type prediction", "Now we will ask you for an output folder"):
        return
    output_folder = dialog.prompt_save_file("Output folder", [("*", "Folder")])
    if output_folder is None:
        return
    os.makedirs(output_folder)

    # Save dataset information
    data_structure = action.to_experiment_list_file_structure(window.get_gui_experiment().get_active_tabs())
    with open(os.path.join(output_folder, "Dataset" + list_io.FILES_LIST_EXTENSION), "w") as handle:
        json.dump(data_structure, handle)

    # Save run script
    create_run_script(output_folder, "types_predict")

    # Save config file
    config = ConfigFile("types_predict", folder_name=output_folder)
    config.set("types_model", types_model_path)
    _parse_config(config)
    config.save()

    # Done!
    if dialog.prompt_options("Run folder created", f"The configuration files were created successfully. Please"
                                                   f" run the types_predict script from that directory:\n\n{output_folder}",
                             option_default=DefaultOption.OK, option_1="Open that directory") == 1:
        dialog.open_file(output_folder)



def run_predictions():
    config = ConfigFile("types_predict")
    parsed_config = _parse_config(config)
    config.save_and_exit_if_changed()

    from . import lib_models
    model = lib_models.load_model(parsed_config.model_folder)

    os.makedirs(parsed_config.output_folder, exist_ok=True)
    for i, experiment in enumerate(list_io.load_experiment_list_file(parsed_config.tracking_input_file)):
        print(f"Working on {experiment.name}...")
        _predict_organoid(experiment, model)
        io.save_data_to_json(experiment,
                             os.path.join(parsed_config.output_folder, f"{i + 1}. {experiment.name.get_save_name()}.{io.FILE_EXTENSION}"))

