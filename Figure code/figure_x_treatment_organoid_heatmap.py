from typing import Iterable

import anndata
import numpy
import pandas
import scanpy.plotting
import scanpy.preprocessing
from anndata import AnnData
from matplotlib import pyplot as plt

import lib_data
import lib_figures
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

_INPUT_FILE = r"../../Data/Testing data - output - treatments.autlist"
_TREATMENT_NAME_TRANSLATION = {
    "control": "Control",
    "dapt chir": "+DAPT +CHIR",
    "EN": "-Rspondin",
    "chir vpa": "+CHIR +VPA"
}


def main():
    adata = _get_adata(list_io.load_experiment_list_file(_INPUT_FILE))
    adata = lib_figures.standard_preprocess(adata)

    # Get mean expression for each organoid
    organoid_list = list(adata.obs.organoid.unique())
    treatment_list = []
    mean_table = numpy.zeros((len(organoid_list), len(adata.var_names)), dtype=numpy.float32)
    for i, organoid in enumerate(organoid_list):
        mean_table[i, :] = adata[adata.obs.organoid == organoid].X.mean(axis=0)
        treatment = adata.obs[adata.obs.organoid == organoid].treatment.iloc[0]
        treatment_list.append(treatment)

    # Create mean_adata with mean expression for each organoid
    mean_adata = scanpy.AnnData(X=mean_table)
    mean_adata.var_names = adata.var_names
    mean_adata.obs_names = organoid_list
    mean_adata.obs["treatment"] = pandas.Categorical(treatment_list)

    # Stylize the variable names
    mean_adata.var_names = [lib_figures.style_variable_name(var_name) for var_name in mean_adata.var_names]

    # Sort genes by their mean expression across all treatments
    mean_adata = mean_adata[:, mean_adata.X.mean(axis=0).argsort()]

    # Reorder the treatments
    mean_adata.obs["treatment"] = pandas.Categorical(mean_adata.obs["treatment"], categories=_TREATMENT_NAME_TRANSLATION.values())

    # Plot the heatmap
    _ = lib_figures.new_figure()
    mean_adata.uns["treatment_colors"] = ["#dfe6e9", "#74b9ff", "#ff7675", "#fdcb6e"]
    scanpy.plotting.heatmap(mean_adata, var_names=mean_adata.var_names, groupby="treatment", cmap="RdBu_r",
                            swap_axes=True, vmin=-2, vmax=2, figsize=(3, 4))
    plt.show()


def _get_adata(experiments: Iterable[Experiment]) -> anndata.AnnData:
    """Gets the raw features, so that we can create a heatmap."""
    data_array = list()
    cell_type_list = list()
    organoid_list = list()
    treatment_list = list()
    cell_names = list()
    # Collect position data for last 10 time points of each experiment
    for experiment in experiments:
        print("Loading", experiment.name)
        treatment = experiment.name.get_name().split("-")[0]
        treatment = _TREATMENT_NAME_TRANSLATION.get(treatment, treatment)
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
                    cell_names.append(
                        f"{experiment.name}-t{position.time_point_number()}-{int(position.x)}-{int(position.y)}-{int(position.z)}")
                    treatment_list.append(treatment)
    data_array = numpy.array(data_array, dtype=numpy.float32)
    adata = AnnData(data_array)
    adata.var_names = lib_data.STANDARD_METADATA_NAMES
    adata.obs_names = cell_names
    adata.obs["cell_type"] = pandas.Categorical(cell_type_list)
    adata.obs["organoid"] = pandas.Categorical(organoid_list)
    adata.obs["treatment"] = pandas.Categorical(treatment_list)
    return adata


if __name__ == "__main__":
    main()
