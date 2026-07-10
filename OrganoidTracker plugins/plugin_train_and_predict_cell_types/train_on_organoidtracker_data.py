import json
import os
from collections import defaultdict
from typing import NamedTuple, List, Dict

import anndata
import numpy
import scanpy
import scipy
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from numpy import ndarray

from organoid_tracker.config import ConfigFile, config_type_int
from organoid_tracker.core import UserError, TimePoint
from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import list_io
from organoid_tracker.util.run_script_creator import create_run_script
from . import lib_data
from . import lib_models


class _ResultTally:
    input_output: lib_models.ModelInputOutput
    confusion_matrix_counts: ndarray
    accuracies: list[float]

    def __init__(self, input_output: lib_models.ModelInputOutput):
        self.input_output = input_output
        self.confusion_matrix_counts = numpy.zeros((len(input_output.cell_type_mapping), len(input_output.cell_type_mapping)), dtype=numpy.uint32)
        self.accuracies = list()

    def post_results(self, ground_truth: ndarray, predicted: ndarray):
        self.confusion_matrix_counts += sklearn.metrics.confusion_matrix(ground_truth, predicted,
                                                                         labels=numpy.arange(len(self.input_output.cell_type_mapping))).astype(numpy.uint32)
        self.accuracies.append(numpy.mean(ground_truth == predicted))

    def save_confusion_matrix(self, output_folder: str):
        confusion_matrix_prediction_scaled = self.confusion_matrix_counts / self.confusion_matrix_counts.sum(axis=1, keepdims=True)

        os.makedirs(output_folder, exist_ok=True)
        figure = plt.figure()
        ax = figure.gca()
        mappable = ax.imshow(confusion_matrix_prediction_scaled, cmap="Blues", vmin=0, vmax=1)
        for x in range(len(self.input_output.cell_type_mapping)):
            for y in range(len(self.input_output.cell_type_mapping)):
                ax.text(x, y, str(self.confusion_matrix_counts[y, x]), ha="center", va="center",
                        color="black" if confusion_matrix_prediction_scaled[y, x] < 0.5 else "white")

        ax.set_xticks(numpy.arange(len(self.input_output.cell_type_mapping)))
        ax.set_xticklabels(self.input_output.cell_type_mapping, rotation=90)
        ax.set_yticks(numpy.arange(len(self.input_output.cell_type_mapping)))
        ax.set_yticklabels(self.input_output.cell_type_mapping)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        figure.suptitle(f"Overall accuracy: {numpy.mean(self.accuracies) * 100:.1f}%")
        ax.set_title(f"Numbers are cell counts, summed over {len(self.accuracies)} rounds of cross-validation.", fontsize=8)
        figure.colorbar(mappable, ax=ax, label=f"Accuracy")
        figure.tight_layout()
        figure.savefig(os.path.join(output_folder, "confusion matrix.pdf"), dpi=300)


class _ParsedConfig(NamedTuple):
    model_folder: str
    tracking_input_file: str
    k_folds: int


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
    model_folder = config.get_or_default("model_folder", "Trained model/",
                                          comment="Path to the folder that will contain the trained model.")
    k_folds = config.get_or_default("k_folds", 5, comment="Number of folds K to use for K-fold cross-validation.", type=config_type_int)
    return _ParsedConfig(
        tracking_input_file=dataset_path,
        model_folder=model_folder,
        k_folds=k_folds
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


def _make_heatmap(x_values: numpy.ndarray, cell_types: list[str], output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    adata = anndata.AnnData(x_values)
    adata.var_names = lib_data.STANDARD_METADATA_NAMES
    adata.obs["cell_type_training"] = cell_types

    scanpy.preprocessing.log1p(adata)
    scanpy.preprocessing.scale(adata)

    # Group similar var names together
    clustering_variables_result = scipy.cluster.hierarchy.linkage(adata.X.T, method="average")
    reordering = scipy.cluster.hierarchy.leaves_list(clustering_variables_result)
    var_names = numpy.array(adata.var_names)
    var_names = var_names[reordering]

    scanpy.plotting.heatmap(adata,
                            var_names=var_names,
                            groupby="cell_type_training",
                            cmap="bwr",
                            swap_axes=True,
                            vmin=-3,
                            vmax=3,
                            show=False)
    figure = plt.gcf()
    figure.savefig(os.path.join(output_folder, "heatmap.pdf"), dpi=300)
    plt.close(figure)


def run_training():
    config = ConfigFile("types_train")
    parsed_config = _parse_config(config)
    config.save_and_exit_if_changed()

    from . import lib_models

    os.makedirs(parsed_config.model_folder, exist_ok=True)

    # Get input parameter names
    input_names = lib_data.STANDARD_METADATA_NAMES

    # Collect position data for last 10 time points of each experiment
    x_values = list()
    cell_types = list()

    for i, experiment in enumerate(list_io.load_experiment_list_file(parsed_config.tracking_input_file)):
        print("Loading", experiment.name)
        cell_type_mapping = lib_data.get_cell_type_mapping(experiment)
        positions = experiment.positions

        last_time_point_number = experiment.positions.last_time_point_number()
        time_points = [TimePoint(last_time_point_number), TimePoint(last_time_point_number - 5),
                       TimePoint(last_time_point_number - 10), TimePoint(last_time_point_number - 15)]
        for time_point in time_points:
            for position in positions.of_time_point(time_point):
                position_data_array = lib_data.get_data_array(positions, position, input_names)
                cell_type = positions.get_position_data(position, "type")
                if not isinstance(cell_type, str):
                    continue
                cell_type_training = cell_type_mapping.get(cell_type)
                if position_data_array is not None and cell_type_training is not None:
                    x_values.append(position_data_array)
                    cell_types.append(cell_type_training)

    x_values = numpy.array(x_values)
    cell_types_used = list(set(cell_types))
    cell_types_used.sort()
    class_weights = _calculate_class_weights(cell_types, cell_types_used)
    y_values = numpy.array([cell_types_used.index(cell_type) for cell_type in cell_types])
    input_output = lib_models.ModelInputOutput(cell_type_mapping=cell_types_used, input_mapping=input_names)

    # Heatmap of input output
    graphs_folder = os.path.join(parsed_config.model_folder, "Graphs")
    _make_heatmap(x_values, cell_types, graphs_folder)

    # K-fold cross-validation
    kfold = KFold(n_splits=parsed_config.k_folds, shuffle=True)
    results = _ResultTally(input_output)
    for train_indices, test_indices in kfold.split(x_values, y_values):
        # Train the model
        model = lib_models.build_shallow_model(input_output, x_values[train_indices], hidden_neurons=0)
        model.fit(x_values[train_indices], y_values[train_indices], epochs=1, class_weights=class_weights)

        # Evaluate the model
        results.post_results(
            y_values[test_indices],
            numpy.argmax(model.predict(x_values[test_indices]), axis=1))

    # Save the model (arbitrarily saving the last model trained)
    model.save(parsed_config.model_folder)

    # Save the results
    results.save_confusion_matrix(graphs_folder)
