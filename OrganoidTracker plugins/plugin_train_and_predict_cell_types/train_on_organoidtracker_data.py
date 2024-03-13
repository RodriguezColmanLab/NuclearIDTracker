import json
import os
from collections import defaultdict
from typing import NamedTuple, List, Dict

import numpy

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError, TimePoint
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
    output_folder: str
    tracking_input_file: str


def _calculate_class_weights(cell_types: List[str], cell_type_by_id: List[str]) -> Dict[int, float]:
    counts_by_type = defaultdict(lambda: 0)
    for cell_type in cell_types:
        counts_by_type[cell_type] += 1

    max_count = max(counts_by_type.values())
    weights_by_type = dict()
    for cell_type, count in counts_by_type.items():
        weights_by_type[cell_type_by_id.index(cell_type)] = min(max_count / count, 2)

    return weights_by_type


def _parse_config(config: ConfigFile) -> _ParsedConfig:
    """Parses the config, and adds any missing comments or default settings."""
    dataset_path = config.get_or_default("dataset_to_segment", "Dataset" + list_io.FILES_LIST_EXTENSION,
                                         comment="Path to the dataset that you are using.")
    output_folder = config.get_or_default("output_folder", "Trained model/",
                                          comment="Path to the folder that will contain the trained model.")
    return _ParsedConfig(
        tracking_input_file=dataset_path,
        output_folder=output_folder
    )


def create_training_script(window: Window):
    """Creates a training script in the GUI. Called from get_menu_items()."""
    # Check open experiments
    experiment_count = 0
    for experiment in window.get_active_experiments():
        if "ellipticity" not in experiment.position_data.get_data_names_and_types():
            raise UserError("Missing extracted parameters", f"You haven't extracted the shape parameters yet from the"
                                                            f" CellPose segmentation for the experiment"
                                                            f" \"{experiment.name}\". Please do so first.")
        experiment_count += 1
    if experiment_count <= 1:
        if not dialog.prompt_confirmation("Cell type training", "You should use multiple experiments to"
                                                                " train for cell types. Tip: make sure that the"
                                                                " experiment selector on the top right is set to"
                                                                " \"All\"."):
            return

    # Creates an output folder
    if not dialog.popup_message_cancellable("Cell type training", "We will ask you for a folder where the"
                                                                  " training files will appear."):
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
    create_run_script(output_folder, "types_train")

    # Save config file
    config = ConfigFile("types_train", folder_name=output_folder)
    _parse_config(config)
    config.save()

    # Done!
    if dialog.prompt_options("Run folder created", f"The configuration files were created successfully. Please"
                                                   f" run the types_train script from that directory:\n\n{output_folder}",
                             option_default=DefaultOption.OK, option_1="Open that directory") == 1:
        dialog.open_file(output_folder)


def run_training():
    config = ConfigFile("types_train")
    parsed_config = _parse_config(config)
    config.save_and_exit_if_changed()

    from . import lib_models

    os.makedirs(parsed_config.output_folder, exist_ok=True)

    # Get input parameter names
    input_names = lib_data.STANDARD_METADATA_NAMES

    # Collect position data for last 10 time points of each experiment
    data_array = list()
    cell_type_training_list = list()

    for i, experiment in enumerate(list_io.load_experiment_list_file(parsed_config.tracking_input_file)):
        print("Loading", experiment.name)
        position_data = experiment.position_data

        last_time_point_number = experiment.positions.last_time_point_number()
        time_points = [TimePoint(last_time_point_number), TimePoint(last_time_point_number - 5),
                       TimePoint(last_time_point_number - 10), TimePoint(last_time_point_number - 15)]
        for time_point in time_points:
            for position in experiment.positions.of_time_point(time_point):
                position_data_array = lib_data.get_data_array(position_data, position, input_names)
                cell_type = position_data.get_position_data(position, "type")
                cell_type_training = lib_data.convert_cell_type(cell_type)
                if position_data_array is not None and cell_type_training != "NONE":
                    data_array.append(position_data_array)
                    cell_type_training_list.append(cell_type_training)

    data_array = numpy.array(data_array)
    cell_types_used = list(set(cell_type_training_list))
    class_weights = _calculate_class_weights(cell_type_training_list, cell_types_used)
    cell_type_training_list = numpy.array([cell_types_used.index(cell_type) for cell_type in cell_type_training_list])
    input_output = lib_models.ModelInputOutput(cell_type_mapping=cell_types_used, input_mapping=input_names)

    # Train the model
    model = lib_models.build_shallow_model(input_output, data_array, hidden_neurons=0)
    model.fit(data_array, cell_type_training_list, epochs=1, class_weights=class_weights)

    # Save the model
    model.save(parsed_config.output_folder)
