import numpy
import scanpy
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import List, NamedTuple

import lib_data
import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import intensity_calculator

_LDA_FILE = "../../Data/all_data.h5ad"
_EXPERIMENTS_FILE = "../../Data/Testing data - output - CD24 and Lgr5.autlist"
_LGR5_AVERAGING_WINDOW_H = 4
_INTENSITY_KEY_LGR5 = "intensity_lgr5"
_INTENSITY_KEY_CD24 = "intensity_cd24"


class _Points(NamedTuple):
    x_values: ndarray
    y_values: ndarray
    lgr5_values: ndarray
    cd24_values: ndarray
    stemness_values: ndarray
    panethness_values: ndarray
    enterocyteness_values: ndarray


def _create_colormap(cell_type_name: str) -> LinearSegmentedColormap:
    dark_color = {
        "STEM": "#337A0D",
        "PANETH": "#AF422A",
        "ENTEROCYTE": "#194160"
    }[cell_type_name]
    mid_color = lib_figures.CELL_TYPE_PALETTE[cell_type_name]
    light_color = "#F4F8F9"
    return LinearSegmentedColormap.from_list("custom_" + cell_type_name, [light_color, mid_color, dark_color])


def main():
    # Load LGR5 and CD24 data
    experiments = list(list_io.load_experiment_list_file(_EXPERIMENTS_FILE))

    # Loading and preprocessing
    adata = scanpy.read_h5ad(_LDA_FILE)
    adata = lib_figures.standard_preprocess(adata)

    # Do the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(adata.X, adata.obs["cell_type_training"])

    # Get the plotting points
    points = _extract_lda_data(adata, lda, experiments)

    # Make a scatter plot
    figure = plt.figure(figsize=(7, 4.5))
    (ax_stemness, ax_panethness, ax_enterocyteness), (ax_lgr5, ax_cd24, ax_empty) = figure.subplots(2, 3, sharex=True, sharey=True)

    # Plot stemness
    ax_stemness.scatter(points.x_values, points.y_values, c=points.stemness_values,
                        cmap=_create_colormap("STEM"), vmin=0, vmax=0.66, s=2, lw=0)
    ax_stemness.set_title("Stem probability")

    # Plot Panethness
    ax_panethness.scatter(points.x_values, points.y_values, c=points.panethness_values,
                          cmap=_create_colormap("PANETH"), vmin=0, vmax=0.66, s=2, lw=0)
    ax_panethness.set_title("Paneth probability")

    # Plot Enterocyteness
    ax_enterocyteness.scatter(points.x_values, points.y_values, c=points.enterocyteness_values,
                              cmap=_create_colormap("ENTEROCYTE"), vmin=0, vmax=1, s=2, lw=0)
    ax_enterocyteness.set_title("Enterocyte probability")

    # Plot LGR5
    min_lgr5 = numpy.percentile(points.lgr5_values, 5)
    max_lgr5 = numpy.percentile(points.lgr5_values, 95)
    ax_lgr5.scatter(points.x_values, points.y_values, c=points.lgr5_values, cmap="Greens", vmin=min_lgr5, vmax=max_lgr5, s=2, lw=0)
    ax_lgr5.set_title("Lgr5 intensity")

    # Plot CD24
    min_cd24 = numpy.percentile(points.cd24_values, 5)
    max_cd24 = numpy.percentile(points.cd24_values, 95)
    ax_cd24.scatter(points.x_values, points.y_values, c=points.cd24_values, cmap="Reds", vmin=min_cd24, vmax=max_cd24, s=2, lw=0)
    ax_cd24.set_title("CD24 intensity")

    # Hide the empty plot
    ax_empty.axis("off")

    # Add colorbars
    figure.colorbar(ax_stemness.collections[0], ax=ax_stemness)
    figure.colorbar(ax_panethness.collections[0], ax=ax_panethness)
    figure.colorbar(ax_enterocyteness.collections[0], ax=ax_enterocyteness,)
    figure.colorbar(ax_lgr5.collections[0], ax=ax_lgr5)
    figure.colorbar(ax_cd24.collections[0], ax=ax_cd24)

    figure.tight_layout()
    plt.show()


def _plot_lda(ax: Axes, lda: LinearDiscriminantAnalysis, adata: AnnData):
    plot_coords = lda.transform(adata.X)

    # Plot the LDA
    used_cell_types = adata.obs["cell_type_training"].array.categories
    for cell_type in used_cell_types:
        depth = -3 if cell_type == "NONE" else 0
        mask = adata.obs["cell_type_training"] == cell_type
        ax.scatter(plot_coords[mask, 0], -plot_coords[mask, 1],
                   s=20, lw=0, zorder=depth,
                   color=lib_figures.CELL_TYPE_PALETTE[cell_type], label=lib_figures.style_cell_type_name(cell_type))
    ax.set_title("Linear Discriminant Analysis")
    ax.legend()


def _extract_lda_data(adata: AnnData, lda: LinearDiscriminantAnalysis, experiments: List[Experiment]) -> _Points:
    input_names = list(adata.var_names)

    all_position_data = list()
    all_intensities_lgr5 = list()
    all_intensities_cd24 = list()
    all_stemness = list()
    all_panethness = list()
    all_enterocyteness = list()

    for experiment in experiments:
        stem_index = experiment.global_data.get_data("ct_probabilities").index("STEM")
        paneth_index = experiment.global_data.get_data("ct_probabilities").index("PANETH")
        enterocyte_index = experiment.global_data.get_data("ct_probabilities").index("ENTEROCYTE")

        for time_point in experiment.time_points():
            for position in experiment.positions.of_time_point(time_point):
                data_array = lib_data.get_data_array(experiment.position_data, position, input_names)
                if data_array is None or numpy.any(numpy.isnan(data_array)):
                    continue

                intensity_lgr5 = intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=_INTENSITY_KEY_LGR5)
                intensity_cd24 = intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=_INTENSITY_KEY_CD24)
                if intensity_lgr5 is None or intensity_cd24 is None:
                    continue

                ct_probabilities = experiment.position_data.get_position_data(position, "ct_probabilities")
                if ct_probabilities is None:
                    continue

                all_position_data.append(data_array)
                all_intensities_lgr5.append(intensity_lgr5)
                all_intensities_cd24.append(intensity_cd24)
                all_stemness.append(ct_probabilities[stem_index])
                all_panethness.append(ct_probabilities[paneth_index])
                all_enterocyteness.append(ct_probabilities[enterocyte_index])

    adata_experiments = AnnData(numpy.array(all_position_data))
    adata_experiments.var_names = input_names
    adata_experiments = lib_figures.standard_preprocess(adata_experiments, filter=False, scale=False)
    adata_experiments.X -= numpy.array(adata.var["mean"])
    adata_experiments.X /= numpy.array(adata.var["std"])

    plot_coords = lda.transform(adata_experiments.X)
    return _Points(x_values=plot_coords[:, 0], y_values=plot_coords[:, 1],
                   lgr5_values=numpy.array(all_intensities_lgr5),
                   cd24_values=numpy.array(all_intensities_cd24),
                   stemness_values=numpy.array(all_stemness),
                   panethness_values=numpy.array(all_panethness),
                   enterocyteness_values=numpy.array(all_enterocyteness))


if __name__ == "__main__":
    main()
