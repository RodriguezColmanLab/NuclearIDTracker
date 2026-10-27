from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy
import pandas
import scanpy
import scanpy.preprocessing
import sklearn.metrics
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from numpy import ndarray
from sklearn.model_selection import KFold
from lib_models import build_shallow_model, ModelInputOutput
import lib_figures

_NUM_FOLDS = 5
_TRAINING_DATA_FILE = "../../Data/all_data.h5ad"
_COLORMAP = LinearSegmentedColormap.from_list("red_white_blue", ["#3434b4", "#ffffff", "#5db263"])

class _ConfusionMatrix(NamedTuple):
    cell_types: List[str]
    confusion_matrix: ndarray

    def accuracy(self) -> float:
        """Returns the overall accuracy of the model based on the confusion matrix."""
        correct_predictions = numpy.diagonal(self.confusion_matrix).sum()
        total_predictions = self.confusion_matrix.sum()
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

    def accuracy_for_cell_type(self, cell_type: str) -> float:
        """Returns the accuracy for a specific cell type based on the confusion matrix."""
        cell_type_index = self.cell_types.index(cell_type)
        correct_predictions = self.confusion_matrix[cell_type_index, cell_type_index]
        total_predictions = numpy.sum(self.confusion_matrix[cell_type_index, :])
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0


class _AblationStudyResults:
    """Stores the effect of removing a parameter on the model's performance."""

    baseline: _ConfusionMatrix

    _results_by_removed_parameter: dict[str, _ConfusionMatrix]

    def __init__(self, baseline: _ConfusionMatrix):
        self.baseline = baseline
        self._results_by_removed_parameter = dict()

    def add_result(self, removed_parameter: str, confusion_matrix: _ConfusionMatrix):
        self._results_by_removed_parameter[removed_parameter] = confusion_matrix

    def get_top_n_parameters(self, n: int) -> List[Tuple[str, _ConfusionMatrix]]:
        """Returns the top n parameters that had the most negative effect on performance when removed."""
        sorted_results = sorted(self._results_by_removed_parameter.items(),
                                key=lambda item: item[1].accuracy())
        return sorted_results[:n]


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


def main():
    comparison = _evaluate_model()
    _show_ablation_study_results(comparison)


def _show_ablation_study_results(comparison: _AblationStudyResults):
    figure = lib_figures.new_figure(size=(5, 3))
    ax = figure.gca()

    n_parameters = 10
    row_count = 1 + n_parameters  # Top row is for baseline results
    column_count = 1 + len(comparison.baseline.cell_types)

    baseline_accuracy = comparison.baseline.accuracy()

    accuracy_image = numpy.zeros((row_count, column_count))
    changes_image = numpy.zeros((row_count, column_count))

    # Fill in the baseline accuracy row
    accuracy_image[0, 0] = baseline_accuracy * 100
    for j, cell_type in enumerate(comparison.baseline.cell_types):
        accuracy_image[0, j + 1] = comparison.baseline.accuracy_for_cell_type(cell_type) * 100

    # Fill in the rows for ablated parameters
    top_parameters = comparison.get_top_n_parameters(n_parameters)
    for i, (parameter, confusion_matrix) in enumerate(top_parameters):
        overall_accuracy = (confusion_matrix.accuracy() / baseline_accuracy - 1) * 100
        accuracy_image[i + 1, 0] = confusion_matrix.accuracy() * 100
        changes_image[i + 1, 0] = overall_accuracy
        for j, cell_type in enumerate(comparison.baseline.cell_types):
            baseline_accuracy = comparison.baseline.accuracy_for_cell_type(cell_type)
            ablated_accuracy = confusion_matrix.accuracy_for_cell_type(cell_type)
            cell_type_accuracy = (ablated_accuracy / baseline_accuracy - 1) * 100

            accuracy_image[i + 1, j + 1] = confusion_matrix.accuracy_for_cell_type(cell_type) * 100
            changes_image[i + 1, j + 1] = cell_type_accuracy

    ax.imshow(changes_image, cmap=_COLORMAP, vmin=-12, vmax=12, aspect=1/3)

    # Draw percentage text on each cell
    for i in range(accuracy_image.shape[0]):
        for j in range(accuracy_image.shape[1]):
            accuracy_change = changes_image[i, j]
            accuracy = accuracy_image[i, j]
            text = f"{accuracy:.1f}%"
            ax.text(j, i, text, ha="center", va="center", color="white" if accuracy_change < -5 else "black")

    ax.set_xticks(numpy.arange(column_count))
    ax.set_xticklabels(["overall"] + [lib_figures.style_cell_type_name(cell_type) for cell_type in comparison.baseline.cell_types], rotation=45, ha="right")
    ax.set_yticks(numpy.arange(row_count))
    ax.set_yticklabels(["baseline"] + [lib_figures.style_variable_name(parameter) for parameter, _ in top_parameters])
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Removed parameter")
    ax.axvline(x=0.5, color="black", linewidth=1)
    ax.axhline(y=0.5, color="black", linewidth=1)
    figure.tight_layout()
    plt.show()


def _evaluate_model() -> _AblationStudyResults:
    adata = scanpy.read_h5ad(_TRAINING_DATA_FILE)
    adata = adata[adata.obs["cell_type_training"] != "NONE"]  # These ones have an unclear training cell type
    cell_types = list(adata.obs["cell_type_training"].array.categories)

    baseline_confusion_matrix = _calculate_accuracy(adata, cell_types)
    overall_results = _AblationStudyResults(baseline_confusion_matrix)
    for parameter_to_ablate in adata.var_names:
        adata_ablated = adata.copy()

        # Remove the parameter from the data
        adata_ablated = adata_ablated[:, adata_ablated.var_names != parameter_to_ablate]

        confusion_matrix_folds = _calculate_accuracy(adata_ablated, cell_types)
        overall_results.add_result(parameter_to_ablate, confusion_matrix_folds)
    return overall_results


def _calculate_accuracy(adata: AnnData, cell_types: list[str]) -> _ConfusionMatrix:
    input_output = ModelInputOutput(cell_type_mapping=list(cell_types), input_mapping=list(adata.var_names))

    x_values = adata.X
    y_values = numpy.array(adata.obs["cell_type_training"].cat.codes.array)
    random = numpy.random.RandomState(1234)
    kfold = KFold(n_splits=_NUM_FOLDS, shuffle=True, random_state=random)
    weights_train = _calculate_class_weights(adata.obs["cell_type_training"])
    confusion_matrix_folds: _ConfusionMatrix | None = None
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
        if confusion_matrix_folds is None:
            confusion_matrix_folds = _ConfusionMatrix(cell_types=cell_types, confusion_matrix=confusion_matrix)
        else:
            existing_array = confusion_matrix_folds.confusion_matrix
            existing_array += confusion_matrix

    if confusion_matrix_folds is None:
        raise ValueError("No folds were processed, cannot compute confusion matrix.")
    return confusion_matrix_folds


if __name__ == "__main__":
    main()
