import anndata
import numpy
import pandas
import scanpy.preprocessing
import sklearn.metrics
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Tuple, Iterable

import lib_data
import lib_figures
from lib_models import build_shallow_model, ModelInputOutput
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_NUM_FOLDS = 5
_DATA_FILE_PREDICTED = "../../Data/Testing data - predictions - treatments - fixed.autlist"
_DATA_FILE_STAINING = "../../Data/Immunostaining conditions.autlist"


_TRAINING_CELL_TYPES = {
    None: "DOUBLE_NEGATIVE",
    "PANETH": "PANETH",
    "ENTEROCYTE": "KRT20_POSITIVE"
}

_TREATMENT_TRANSLATION = {
    "control": "Control",
    "dapt chir": "+DAPT +CHIR",
    "EN": "-Rspondin",
    "chir vpa R2": "+CHIR +VPA"
}


def _calculate_class_weights(cell_types: pandas.Series) -> Dict[int, float]:
    counts_by_type = dict()

    array = cell_types.cat.codes.array
    for i in numpy.unique(array):
        count = numpy.count_nonzero(array == i)
        counts_by_type[i] = count

    max_count = max(counts_by_type.values())
    weights_by_type = dict()
    for i, count in counts_by_type.items():
        weights_by_type[i] = min(max_count / count, 2)

    return weights_by_type


class _SingleParameterResults:
    """Holds the results of a single parameter set, like for 20 hidden neurons and 5 epochs."""
    confusion_matrix: Optional[ndarray]

    def __init__(self, *, confusion_matrix: Optional[ndarray] = None):
        self.confusion_matrix = confusion_matrix

    def append(self, result: "_SingleParameterResults"):
        if result.confusion_matrix is not None:
            if self.confusion_matrix is None:
                self.confusion_matrix = result.confusion_matrix
            else:
                self.confusion_matrix += result.confusion_matrix


def main():
    adata_all = _load_adata()
    for treatment in adata_all.obs["treatment"].cat.categories:
        adata = adata_all[adata_all.obs["treatment"] == treatment, :]

        cell_types, results = _evaluate_model(adata)
        confusion_matrix = results.confusion_matrix
        fraction_correct = numpy.diagonal(confusion_matrix).sum() / confusion_matrix.sum()
        space_for_bars = len(cell_types) + 0.75

        figure = lib_figures.new_figure()
        ax: Axes = figure.gca()
        for i, cell_type_immunostaining in enumerate(cell_types):
            x_values = [i * space_for_bars + j for j in range(len(cell_types))]
            y_values = [confusion_matrix[i, j] / numpy.sum(confusion_matrix[i]) * 100 for j in range(len(cell_types))]
            ax.bar(x_values, y_values, color=[lib_figures.CELL_TYPE_PALETTE[cell_types[j]] for j in range(len(cell_types))],
                   width=1, align="edge")
            for x, y, cell_type in zip(x_values, y_values, cell_types):
                if cell_type == cell_type_immunostaining:
                    ax.text(x + 0.5, y - 1, lib_figures.style_cell_type_name(cell_type), rotation=90, horizontalalignment="center", verticalalignment="top",
                            color="white")
                    # Highlight the bar
                    # ax.bar([x], [y], color=[(0, 0, 0, 0)], width=1, align="edge", linewidth=3, edgecolor="black")
                else:
                    ax.text(x + 0.5, y + 3, lib_figures.style_cell_type_name(cell_type), rotation=90, horizontalalignment="center")

        ax.set_xticks([space_for_bars * i + 2 for i in range(len(cell_types))])
        ax.set_xticklabels([lib_figures.style_cell_type_name(name) for name in cell_types])
        ax.set_xlabel("Cell type from immunostaining")
        ax.set_ylabel("Predicted types (%)")
        ax.set_title(f"[{treatment}] Accuracy: {fraction_correct * 100:.1f}%")
        ax.set_ylim(0, 100)

        plt.show()


def _load_adata() -> scanpy.AnnData:
    cell_types_staining = _get_staining_table(list_io.load_experiment_list_file(_DATA_FILE_STAINING))
    measurement_data_predicted = _get_adata_predictions(list_io.load_experiment_list_file(_DATA_FILE_PREDICTED))
    measurement_data_predicted = lib_figures.standard_preprocess(measurement_data_predicted)

    # Remove predicted cell types, they are from the old model. We only use this dataset for the measurements of
    # the nuclei.
    del measurement_data_predicted.obs["cell_type"]

    # Add the immunostaining to the predicted data, so that we have one dataset with all information
    combined = pandas.merge(measurement_data_predicted.obs, cell_types_staining, left_index=True, right_index=True,
                            how="left")
    measurement_data_predicted.obs["cell_type_training"] = combined["cell_type_training"]

    return measurement_data_predicted


def _get_adata_predictions(experiments: Iterable[Experiment]) -> anndata.AnnData:
    """Gets the raw features, so that we can create a heatmap. Cells are indexed by _get_cell_key."""
    data_array = list()
    cell_type_list = list()
    organoid_list = list()
    treatment_list = list()
    cell_names = list()
    # Collect position data for last 10 time points of each experiment
    for experiment in experiments:
        print("Loading", experiment.name)
        treatment = experiment.name.get_name().split("-")[0]
        treatment = _TREATMENT_TRANSLATION.get(treatment, treatment)
        position_data = experiment.position_data

        last_time_point_number = experiment.positions.last_time_point_number()
        time_points = [TimePoint(last_time_point_number), TimePoint(last_time_point_number - 5),
                       TimePoint(last_time_point_number - 10), TimePoint(last_time_point_number - 15)]
        for time_point in time_points:
            for position in experiment.positions.of_time_point(time_point):
                position_data_array = lib_data.get_data_array(position_data, position, lib_data.STANDARD_METADATA_NAMES)
                cell_type = position_data.get_position_data(position, "type")
                if position_data_array is not None and cell_type is not None:
                    data_array.append(position_data_array)
                    cell_type_list.append(cell_type)
                    organoid_list.append(experiment.name.get_name())
                    cell_names.append(_get_cell_key(experiment, position))
                    treatment_list.append(treatment)
    data_array = numpy.array(data_array, dtype=numpy.float32)

    adata = AnnData(data_array)
    adata.var_names = lib_data.STANDARD_METADATA_NAMES
    adata.obs_names = cell_names
    adata.obs["cell_type"] = pandas.Categorical(cell_type_list)
    adata.obs["organoid"] = pandas.Categorical(organoid_list)
    adata.obs["treatment"] = pandas.Categorical(treatment_list)

    return adata


def _get_cell_key(experiment: Experiment, position: Position) -> str:
    return f"{experiment.name}-{int(position.x)}-{int(position.y)}-{int(position.z)}"


def _get_staining_table(experiments: Iterable[Experiment]) -> pandas.DataFrame:
    """Get a dictionary of cell types for each position in each experiment."""

    cell_types_predicted = dict()
    for experiment in experiments:
        print(f"Working on {experiment.name.get_name()}...")
        position_data = experiment.position_data

        for time_point in experiment.positions.time_points():
            for position in experiment.positions.of_time_point(time_point):
                position_type = position_markers.get_position_type(position_data, position)
                position_type = _TRAINING_CELL_TYPES.get(position_type, position_type)
                if position_type is None:
                    continue  # Still none after replacements, skip
                cell_types_predicted[_get_cell_key(experiment, position)] = position_type

    data_frame = pandas.DataFrame.from_dict(cell_types_predicted, orient="index", columns=["cell_type_training"])
    data_frame["cell_type_training"] = pandas.Categorical(data_frame["cell_type_training"])
    return data_frame


def _evaluate_model(adata: anndata.AnnData) -> Tuple[List[str], _SingleParameterResults]:



    cell_types = adata.obs["cell_type_training"].array.categories
    input_output = ModelInputOutput(cell_type_mapping=list(cell_types), input_mapping=list(adata.var_names))
    x_values = adata.X
    y_values = numpy.array(adata.obs["cell_type_training"].cat.codes.array)
    random = numpy.random.RandomState(1234)
    kfold = KFold(n_splits=_NUM_FOLDS, shuffle=True, random_state=random)
    weights_train = _calculate_class_weights(adata.obs["cell_type_training"])
    results = _SingleParameterResults()
    for train_indices, test_indices in kfold.split(x_values, y_values):
        # Build and train the model
        model = build_shallow_model(input_output, x_values[train_indices], hidden_neurons=0)
        model.fit(x_values[train_indices, :], y_values[train_indices],
                  class_weights=weights_train)

        # Evaluate the model
        predictions = model.predict(x_values[test_indices])
        answers = y_values[test_indices]

        confusion_matrix = sklearn.metrics.confusion_matrix(
            answers, numpy.argmax(predictions, axis=1), labels=numpy.arange(len(cell_types)))

        results.append(_SingleParameterResults(confusion_matrix=confusion_matrix))
    return cell_types, results


if __name__ == "__main__":
    main()
