import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy
import pandas
import scanpy
import scanpy.preprocessing
import tensorflow
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.model_selection import KFold

from lib_models import build_model, ModelInputOutput

_NUM_FOLDS = 5
_HIDDEN_NEURONS = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
_EPOCHS = [1, 2, 3, 4, 5, 6]
_OUTPUT_FOLDER = "../Data/Models"
_TRAINING_DATA_FILE = "../Data/all_data.h5ad"


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
    accuracies: List[float]
    confusion_matrix: Optional[ndarray]

    def __init__(self, *, accuracy: Optional[float] = None, confusion_matrix: Optional[ndarray] = None):
        self.accuracies = list()
        if accuracy is not None:
            self.accuracies.append(accuracy)
        self.confusion_matrix = confusion_matrix

    def append(self, result: "_SingleParameterResults"):
        if result.confusion_matrix is not None:
            if self.confusion_matrix is None:
                self.confusion_matrix = result.confusion_matrix
            else:
                self.confusion_matrix += result.confusion_matrix
        self.accuracies += result.accuracies


class _ResultContainer:
    _accuracy_by_hidden_neurons: Dict[Tuple[int, int], _SingleParameterResults]

    def __init__(self):
        self._accuracy_by_hidden_neurons = defaultdict(_SingleParameterResults)

    def post(self, hidden_neurons: int, epochs: int, result: _SingleParameterResults):
        self._accuracy_by_hidden_neurons[(hidden_neurons, epochs)].append(result)

    def get_mean_accuracies(self, hidden_neurons_list: List[int], epochs: int) -> ndarray:
        return numpy.array([numpy.mean(self._accuracy_by_hidden_neurons[(hidden_neurons, epochs)].accuracies)
                            for hidden_neurons in hidden_neurons_list])

    def get_standard_deviations(self, hidden_neurons_list: List[int], epochs: int) -> ndarray:
        return numpy.array([numpy.std(self._accuracy_by_hidden_neurons[(hidden_neurons, epochs)].accuracies, ddof=1)
                            for hidden_neurons in hidden_neurons_list])

    def get_confusion_matrix(self, hidden_neurons: int, epochs: int) -> ndarray:
        return self._accuracy_by_hidden_neurons[(hidden_neurons, epochs)].confusion_matrix


def main():
    tensorflow.config.experimental.set_memory_growth(tensorflow.config.list_physical_devices('GPU')[0], True)

    results = _ResultContainer()

    adata = scanpy.read_h5ad(_TRAINING_DATA_FILE)
    adata = adata[adata.obs["cell_type_training"] != "NONE"]  # These ones have an unclear training cell type

    cell_types = adata.obs["cell_type_training"].array.categories
    input_output = ModelInputOutput(cell_type_mapping=list(cell_types), input_mapping=list(adata.var_names))
    x_values = adata.X
    y_values = numpy.array(adata.obs["cell_type_training"].cat.codes.array)

    for epochs in _EPOCHS:
        for hidden_neurons in _HIDDEN_NEURONS:
            if hidden_neurons == 0 and epochs > 1:
                continue  # Doesn't make sense to test this, as for linear models we don't use epochs
            print(f"Working on epochs {epochs}, neurons {hidden_neurons}...")
            kfold = KFold(n_splits=_NUM_FOLDS, shuffle=True)
            weights_train = _calculate_class_weights(adata.obs["cell_type"])
            saved_a_model = False
            for train_indices, test_indices in kfold.split(x_values, y_values):
                # Build and train a neural network
                model = build_model(input_output, x_values[train_indices], hidden_neurons=hidden_neurons)
                model.fit(x_values[train_indices, :], y_values[train_indices],
                          epochs=epochs, class_weights=weights_train)

                # Evaluate the model
                scores = model.evaluate(x_values[test_indices], y_values[test_indices])
                confusion_matrix = numpy.array(tensorflow.math.confusion_matrix(
                    y_values[test_indices],
                    numpy.argmax(model.predict(x_values[test_indices]), axis=1)))

                results.post(hidden_neurons, epochs, _SingleParameterResults(
                    accuracy=scores["accuracy"], confusion_matrix=confusion_matrix))

                # Save the first model (out of all cross-validated models)
                if not saved_a_model:
                    folder = os.path.join(_OUTPUT_FOLDER, f"epochs-{epochs}-neurons-{hidden_neurons}")
                    os.makedirs(folder, exist_ok=True)
                    model.save(folder)
                    saved_a_model = True

    _plot_results(results)
    print(cell_types)
    print(results.get_confusion_matrix(0, 1))


def _plot_results(results: _ResultContainer):
    figure = plt.figure()
    ax: Axes = figure.gca()
    colors = ["#cee6f9", "#84c2f1", "#3a9de9", "#0877cc", "#054f88", "#032844"]
    for i, epochs in enumerate(_EPOCHS):
        color = colors[i % len(colors)]
        accuracy_means = results.get_mean_accuracies(_HIDDEN_NEURONS, epochs)
        accuracy_stdevs = results.get_standard_deviations(_HIDDEN_NEURONS, epochs)
        ax.plot(_HIDDEN_NEURONS, accuracy_means, linewidth=2, markersize=12, color=color, label=f"{epochs} epochs")
        ax.errorbar(_HIDDEN_NEURONS, accuracy_means, yerr=accuracy_stdevs, color=color, capsize=2)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Neurons in hidden layer")
    ax.set_ylabel("Accuracy (± st.dev.)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
