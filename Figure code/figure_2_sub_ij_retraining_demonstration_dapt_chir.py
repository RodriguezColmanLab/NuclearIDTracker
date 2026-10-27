from typing import NamedTuple

import matplotlib
import numpy
import pandas
import scipy
import sklearn
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegression

import lib_data
import lib_models
import lib_figures
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_GROUND_TRUTH = "../../Data/Immunostaining conditions.autlist"
_DATA_FILE_PREDICTED = "../../Data/Testing data - predictions - treatments - fixed.autlist"

_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list("fixed", ["#ffffff", "#008b8b"])
_BASELINE_MODEL_PATH = "../../Data/Models/epochs-1-neurons-0/"


class _CellKey(NamedTuple):
    experiment_name: str
    position: Position


def _build_ground_truth_cell_type_dict() -> dict[_CellKey, str]:
    """Builds a dictionary that maps each cell to its ground truth cell type."""
    cell_type_dict = dict()
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_GROUND_TRUTH, load_images=False):
        experiment_name = experiment.name.get_name()
        positions = experiment.positions
        for position in positions:
            cell_type = position_markers.get_position_type(positions, position)
            if cell_type is None:
                cell_type = "STEM"  # We regarded double-negative cells as stem cells
            cell_key = _CellKey(experiment_name, position)
            cell_type_dict[cell_key] = cell_type
    return cell_type_dict


def _build_dataset() -> AnnData:
    """Builds a dataset with all features and cell types, for the experiments in _DATA_FILE_PREDICTED.

    adata.X will contain the features,
    adata.obs["predicted_cell_type"] will contain the predicted cell types, and
    adata.obs["ground_truth_cell_type"] will contain the ground truth cell types.
    adata.obs["replicate"] will contain the replicate
    """
    ground_truth_cell_types_lookup = _build_ground_truth_cell_type_dict()

    all_features = list()
    obs_names = list()
    predicted_cell_types = list()
    ground_truth_cell_types = list()
    replicates = list()

    for experiment in list_io.load_experiment_list_file(_DATA_FILE_PREDICTED, load_images=False):

        experiment_name = experiment.name.get_name()
        if not "dapt chir" in experiment_name:
            continue

        positions = experiment.positions

        if "R1" in experiment_name:
            replicate = "R1"
        elif "R2" in experiment_name:
            replicate = "R2"
        else:
            raise ValueError(f"Unknown replicate in experiment name: {experiment_name}")


        for position, predicted_cell_type in positions.find_all_positions_with_data("type"):
            features = lib_data.get_data_array(positions, position, lib_data.STANDARD_METADATA_NAMES)
            if features is None:
                continue
            ground_truth_cell_type = ground_truth_cell_types_lookup.get(_CellKey(experiment_name, position))
            if ground_truth_cell_type is None:
                continue

            predicted_cell_types.append(predicted_cell_type)
            all_features.append(features)
            ground_truth_cell_types.append(ground_truth_cell_type)
            replicates.append(replicate)
            obs_names.append(str(position))

    adata = AnnData(X=numpy.array(all_features))
    adata.var_names = lib_data.STANDARD_METADATA_NAMES
    adata.obs_names = obs_names
    adata.obs["predicted_cell_type"] = pandas.Categorical(predicted_cell_types)
    adata.obs["ground_truth_cell_type"] = pandas.Categorical(ground_truth_cell_types)
    adata.obs["replicate"] = pandas.Categorical(replicates)
    return adata

def main():
    adata = _build_dataset()

    # Train a model on replicate 1 and test on replicate 2
    adata_train = adata[adata.obs["replicate"] == "R1"]
    cell_types = adata.obs["ground_truth_cell_type"].cat.categories.tolist()
    new_model = lib_models.build_shallow_model(lib_models.ModelInputOutput(cell_types, adata_train.var_names.tolist()), adata_train.X, hidden_neurons=0)
    new_model.fit(adata_train.X, adata_train.obs["ground_truth_cell_type"].cat.codes,
                  class_weights=lib_models.calculate_class_weights(adata_train.obs["ground_truth_cell_type"]))

    adata_test = adata[adata.obs["replicate"] == "R2"]
    predictions = new_model.predict(adata_test.X)
    predicted_cell_types = [cell_types[numpy.argmax(prediction)] for prediction in predictions]
    adata_test.obs["updated_predicted_cell_type"] = pandas.Categorical(predicted_cell_types, categories=cell_types)

    # Draw confusion matrices before and after retraining
    figure = lib_figures.new_figure(size=(4.3, 2))
    ax_before, ax_after = figure.subplots(nrows=1, ncols=2, sharey=True)
    _draw_confusion_matrix(ax_before, adata_test.obs["ground_truth_cell_type"], adata_test.obs["predicted_cell_type"])
    _draw_confusion_matrix(ax_after, adata_test.obs["ground_truth_cell_type"], adata_test.obs["updated_predicted_cell_type"])
    figure.tight_layout()
    plt.show()

    # Draw model parameters used
    baseline_model = lib_models.load_model(_BASELINE_MODEL_PATH)
    _draw_coefficients(baseline_model, new_model)

def _draw_confusion_matrix(ax: Axes, cell_types_row: pandas.Series, cell_types_column: pandas.Series):
    cell_types_row_indexed = sorted(cell_types_row.cat.categories)
    cell_types_column_indexed = sorted(cell_types_column.cat.categories)

    confusion_matrix = numpy.zeros((len(cell_types_row_indexed), len(cell_types_column_indexed)), dtype=numpy.uint16)
    for i, cell_type_row in enumerate(cell_types_row_indexed):
        for j, cell_type_column in enumerate(cell_types_column_indexed):
            confusion_matrix[i, j] = ((cell_types_row == cell_type_row) & (cell_types_column == cell_type_column)).sum()
    accuracy = numpy.diagonal(confusion_matrix).sum() / confusion_matrix.sum()

    ax.set_title(f"Accuracy: {accuracy * 100:.1f}%")
    confusion_matrix_scaled = confusion_matrix.astype(float)
    confusion_matrix_scaled /= confusion_matrix_scaled.sum(axis=1, keepdims=True)
    ax.imshow(confusion_matrix_scaled, cmap=_COLORMAP)
    ax.set_xticks(range(len(cell_types_column_indexed)))
    ax.set_yticks(range(len(cell_types_row_indexed)))
    ax.set_xticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in cell_types_column_indexed],
                       rotation=-45, ha="left")
    ax.set_yticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in cell_types_row_indexed])
    ax.set_xlabel("Predicted cell type")
    ax.set_ylabel("Cell type (immunostaining)")

    # Add counts to all cells
    for i in range(len(cell_types_row_indexed)):
        for j in range(len(cell_types_column_indexed)):
            count = confusion_matrix[i, j]
            color = "white" if confusion_matrix_scaled[i, j] > 0.5 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)


def _draw_coefficients(baseline_model: lib_models.OurModel, updated_model: lib_models.OurModel, *, number_of_shown_coefficients: int = 14):
    updated_regressor: LogisticRegression = updated_model.logistic_regression

    baseline_regressor: LogisticRegression = baseline_model.logistic_regression
    baseline_cell_types = baseline_model.get_input_output().cell_type_mapping
    baseline_input_mapping = baseline_model.get_input_output().input_mapping

    input_output = updated_model.get_input_output()

    figure = lib_figures.new_figure(size=(3, 6))
    axes = figure.subplots(nrows=len(input_output.cell_type_mapping), ncols=1, sharex=True, sharey=True)

    for ax, cell_type, coefficients_new in zip(axes, updated_model.get_input_output().cell_type_mapping,
                                               updated_regressor.coef_):
        ax: Axes

        sorting = numpy.argsort(coefficients_new)

        features = [lib_figures.style_variable_name(input_output.input_mapping[i]) for i in sorting]
        coefficients_new = coefficients_new[sorting]

        # Looking up the corresponding coefficients in the baseline model is a bit more tricky,
        # because the indices are different. So we need to look up the correct cell type and input variable indices
        coefficients_baseline = baseline_regressor.coef_[baseline_cell_types.index(cell_type)]
        sorting_baseline = [baseline_input_mapping.index(input_output.input_mapping[i]) for i in sorting]
        coefficients_baseline = coefficients_baseline[sorting_baseline]

        # Calculate correlation between the two sets of coefficients
        correlation_spearman = scipy.stats.spearmanr(coefficients_new, coefficients_baseline).correlation
        ax.set_title(f"{lib_figures.style_cell_type_name(cell_type)} (r={correlation_spearman:.2f})")

        # Restrict to the top N coefficients
        features = numpy.concatenate([features[:number_of_shown_coefficients // 2], features[-number_of_shown_coefficients // 2:]])
        coefficients_new = numpy.concatenate([coefficients_new[:number_of_shown_coefficients // 2], coefficients_new[-number_of_shown_coefficients // 2:]])
        coefficients_baseline = numpy.concatenate([coefficients_baseline[:number_of_shown_coefficients // 2], coefficients_baseline[-number_of_shown_coefficients // 2:]])

        for i in range(len(coefficients_new)):
            alignment = "left"
            x_position = 1.8
            ax.text(x_position, i, features[i], ha=alignment, va="center", in_layout=False, color="#666666")
        ax.barh(y=range(len(coefficients_new)), width=coefficients_new, color="#005757", height=0.3)
        ax.barh(y=numpy.arange(len(coefficients_baseline)) + 0.3, width=coefficients_baseline, color="#8686d2", height=0.3, alpha=0.5)

        # Hide vertical axis line
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left']].set_visible(False)

    axes[-1].set_ylim(-1, number_of_shown_coefficients - 0.5)
    axes[-1].set_xlim(-1.3, 4.7)
    axes[-1].set_xlabel("Coefficient")
    figure.tight_layout(w_pad=0)
    plt.show()



if __name__ == "__main__":
    main()
