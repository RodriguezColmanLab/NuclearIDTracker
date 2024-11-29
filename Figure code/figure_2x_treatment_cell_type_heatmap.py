from typing import Iterable, Dict

import anndata
import numpy
import pandas
import scanpy.plotting
from anndata import AnnData
from matplotlib import pyplot as plt

import lib_data
import lib_figures
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_PREDICTED = "../../Data/Testing data - predictions - treatments - fixed.autlist"
_DATA_FILE_STAINING = "../../Data/Immunostaining conditions.autlist"
_FEATURES_PER_STAINING = 5

# Parameters specific to this script
_SKIPPED_TREATMENTS = ["+CHIR +VPA"]
_USED_STAININGS = ["double-negative", "lysozym-positive", "KRT20-positive"]


_TREATMENT_TRANSLATION = {
    "control": "Control",
    "dapt chir": "+DAPT +CHIR",
    "EN": "-Rspondin",
    "chir vpa R2": "+CHIR +VPA"
}
_PREDICTED_CELL_TYPES = {
    "ENTEROCYTE": "enterocyte",
    "PANETH": "Paneth",
    "STEM": "stem",
    "MATURE_GOBLET": None  # This type gets deleted, since we don't stain for it
}
_STAINING_CELL_TYPES = {
    None: "double-negative",
    "PANETH": "lysozym-positive",
    "ENTEROCYTE": "KRT20-positive"
}


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
        if treatment in _SKIPPED_TREATMENTS:
            continue  # Skip this treatment, as it didn't work

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


def main():
    # Load both datasets (immunostaining and predicted types)
    cell_types_staining = _get_staining_table(list_io.load_experiment_list_file(_DATA_FILE_STAINING))
    measurement_data_predicted = _get_adata_predictions(list_io.load_experiment_list_file(_DATA_FILE_PREDICTED))
    measurement_data_predicted = lib_figures.standard_preprocess(measurement_data_predicted)

    # Add the immunostaining to the predicted data, so that we have one dataset with all information
    combined = pandas.merge(measurement_data_predicted.obs, cell_types_staining, left_index=True, right_index=True,
                            how="left")
    measurement_data_predicted.obs["immunostaining"] = combined["immunostaining"]



    # Find the most variable features for each staining
    most_variable_featurues_by_staining = _find_most_variable_features(measurement_data_predicted)

    var_names = measurement_data_predicted.var_names
    for staining, features in most_variable_featurues_by_staining.items():
        print(f"Most variable features for {staining}:")
        for feature in features:
            print(f"  {var_names[feature]}")

    # Now, build the heatmap with treatments as columns, organoids as subcolumns, stainings as rows,
    # and the most variable features as subrows
    organoids = measurement_data_predicted.obs["organoid"].dtype.categories

    # We only use the lysozym staining with the control and +DAPT +CHIR treatments
    stainings = _USED_STAININGS
    treatments = list(measurement_data_predicted.obs["treatment"].dtype.categories)
    treatments.sort(reverse=True)  # This makes control appear on the left
    heatmap = numpy.zeros((len(stainings) * _FEATURES_PER_STAINING, len(organoids)), dtype=numpy.float32)

    organoids_as_plotted = list()  # Essentially the column names
    treatments_as_plotted = list()  # The treatments, corresponding to organoids_as_plotted
    features_as_plotted = list()  # Essentially the row names
    treatment_separation_lines = list()  # Vertical lines that separate the treatments
    for staining_index, staining in enumerate(stainings):

        # We only want to list each organoid once on the X axis, not once per staining
        organoids_as_plotted.clear()
        treatments_as_plotted.clear()

        # Find which features we're going to plot
        for feature_index in most_variable_featurues_by_staining[staining]:
            features_as_plotted.append(lib_figures.style_variable_name(var_names[feature_index]))

        heatmap_column = 0
        data_subset_staining = measurement_data_predicted[measurement_data_predicted.obs["immunostaining"] == staining]
        for j, treatment in enumerate(treatments):
            data_subset_treatment_staining = data_subset_staining[data_subset_staining.obs["treatment"] == treatment]

            # Get all organoids in this treatment (even if they don't have any cells of the current staining)
            data_subset_treatment = measurement_data_predicted[measurement_data_predicted.obs["treatment"] == treatment]
            organoids_in_treatment = list(set(data_subset_treatment.obs["organoid"]))

            for k, organoid in enumerate(organoids_in_treatment):
                data_subset_organoid = data_subset_treatment_staining[
                    data_subset_treatment_staining.obs["organoid"] == organoid]
                if data_subset_organoid.n_obs > 0:
                    heatmap[staining_index * _FEATURES_PER_STAINING:staining_index * _FEATURES_PER_STAINING
                            + _FEATURES_PER_STAINING, heatmap_column]\
                        = data_subset_organoid.X.mean(axis=0)[most_variable_featurues_by_staining[staining]]
                else:
                    heatmap[staining_index * _FEATURES_PER_STAINING:staining_index * _FEATURES_PER_STAINING
                            + _FEATURES_PER_STAINING, heatmap_column] = numpy.nan
                heatmap_column += 1
                organoids_as_plotted.append(organoid)
                treatments_as_plotted.append(treatment)

            treatment_separation_lines.append(len(organoids_as_plotted) - 0.5)

    # Subtract the control values
    control_indices = numpy.array([treatments_as_plotted[i] == "Control" for i in range(len(treatments_as_plotted))])
    control_features_average = heatmap[:, control_indices].mean(axis=1)
    heatmap -= control_features_average[:, numpy.newaxis]

    # Normalize the heatmap per row using the standard deviation
    # Filter out NaNs, since they would cause the standard deviation to be NaN
    for row in range(heatmap.shape[0]):
        row_data = heatmap[row, :]
        row_data = row_data[~numpy.isnan(row_data)]
        heatmap[row, :] /= row_data.std()

    # Plot the heatmap
    figure = lib_figures.new_figure()
    ax = figure.gca()
    ax.imshow(heatmap, cmap="RdBu_r", vmin=-5, vmax=5)
    ax.set_xticks(numpy.arange(len(organoids_as_plotted)), organoids_as_plotted, rotation=-45,
                  horizontalalignment="left")
    ax.set_yticks(numpy.arange(len(features_as_plotted)), features_as_plotted)
    # Add annotations for the stainings
    for i, staining in enumerate(stainings):
        ax.axhline(i * _FEATURES_PER_STAINING - 0.5, color="black")
        ax.text(len(organoids_as_plotted), i * _FEATURES_PER_STAINING + _FEATURES_PER_STAINING / 2, staining,
                ha="left", va="center", rotation=-45)
    # Add lines for the treatments
    for line in treatment_separation_lines:
        ax.axvline(line, color="black")
    plt.show()


def _find_most_variable_features(measurement_data_predicted: AnnData) -> Dict[str, numpy.ndarray]:
    """Finds the most variable features for each staining across the treatments."""
    stainings = measurement_data_predicted.obs["immunostaining"].dtype.categories
    treatments = measurement_data_predicted.obs["treatment"].dtype.categories

    most_variable_featurues_by_staining = dict()
    for staining_index, staining in enumerate(stainings):
        data_subset_staining = measurement_data_predicted[measurement_data_predicted.obs["immunostaining"] == staining]

        # Calculate the mean expression for each gene per treatment
        mean_table = numpy.zeros((len(treatments), len(data_subset_staining.var_names)), dtype=numpy.float32)
        for j, treatment in enumerate(treatments):
            data_subset_treatment_staining = data_subset_staining[data_subset_staining.obs["treatment"] == treatment]
            mean_table[j, :] = data_subset_treatment_staining.X.mean(axis=0)

        # Find the genes with the most variation across the treatments
        variance = mean_table.var(axis=0)
        variance_sorted = variance.argsort()[::-1]
        most_variable_features = variance_sorted[:_FEATURES_PER_STAINING]
        most_variable_featurues_by_staining[staining] = most_variable_features
    return most_variable_featurues_by_staining


def _get_staining_table(experiments: Iterable[Experiment]) -> pandas.DataFrame:
    """Get a dictionary of cell types for each position in each experiment."""

    cell_types_predicted = dict()
    for experiment in experiments:
        print(f"Working on {experiment.name.get_name()}...")
        position_data = experiment.position_data

        for time_point in experiment.positions.time_points():
            for position in experiment.positions.of_time_point(time_point):
                position_type = position_markers.get_position_type(position_data, position)
                position_type = _STAINING_CELL_TYPES.get(position_type, position_type)
                if position_type is None:
                    continue  # Still none after replacements, skip
                cell_types_predicted[_get_cell_key(experiment, position)] = position_type

    data_frame = pandas.DataFrame.from_dict(cell_types_predicted, orient="index", columns=["immunostaining"])
    data_frame["immunostaining"] = pandas.Categorical(data_frame["immunostaining"])
    return data_frame


if __name__ == "__main__":
    main()
