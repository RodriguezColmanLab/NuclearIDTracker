import math

import matplotlib.colors
import numpy
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
import sklearn.discriminant_analysis
from matplotlib import pyplot as plt

import lib_figures
import lib_data

INPUT_FILE = "../../Data/all_data.h5ad"
_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", ["#ECECEC", "#4054A3"])

def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(INPUT_FILE)
    adata = lib_figures.standard_preprocess(adata, log1p=False, scale=False)

    # Remove cells that we cannot train on
    #adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Print the average of the extent feature for each cell type
    index = list(adata.var_names).index("extent")
    for cell_type in adata.obs["cell_type_training"].cat.categories:
        mask = adata.obs["cell_type_training"] == cell_type
        print(cell_type, numpy.mean(adata.X[mask, index]))

    # Do the LDA
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    adata_scaled = adata.copy()
    scanpy.preprocessing.log1p(adata_scaled)
    scanpy.preprocessing.scale(adata_scaled)
    plot_coords = lda.fit(adata_scaled.X, adata_scaled.obs["cell_type_training"]).transform(adata_scaled.X)

    # Plot the LDA
    features = adata.var_names
    rows = int(math.sqrt(len(features)))
    cols = math.ceil(len(features) / rows)
    figure = lib_figures.new_figure(size=(9, 9))
    axes = numpy.array(figure.subplots(rows, cols, sharex=True, sharey=True)).flatten()

    colorbar = numpy.linspace(0, 1, 50)
    colorbar = colorbar.reshape((colorbar.shape[0], 1))

    for i, feature, ax in zip(range(len(features)), features, axes):
        values = adata.X[:, i]

        if lib_data.should_be_exponential(feature):
            values = numpy.log(values)  # Undo the automatic exponential scaling

        vmin = numpy.percentile(values, 10)
        vmax = numpy.percentile(values, 99)
        ax.scatter(plot_coords[:, 0], plot_coords[:, 1],
                   s=9, lw=0, c=values, vmin=vmin, vmax=vmax, cmap=_COLORMAP)

        plox_x_start, plot_x_end = plot_coords[:, 0].min() - 0.5, plot_coords[:, 0].max() + 0.5
        plot_y_start, plot_y_end = plot_coords[:, 1].min() - 0.5, plot_coords[:, 1].max() + 0.5
        ax.imshow(colorbar, cmap=_COLORMAP, aspect="auto", extent=(plox_x_start * 0.15 + plot_x_end * 0.85,
                                                                   plox_x_start * 0.05 + plot_x_end * 0.95,
                                                                   plot_y_start * 0.95 + plot_y_end * 0.05,
                                                                   plot_y_start * 0.65 + plot_y_end * 0.35))

        vmin_str = f"{vmin:.1f}" if vmin < 50 else str(int(vmin))
        vmax_str = f"{vmax:.1f}" if vmin < 50 else str(int(vmax))
        ax.text(plox_x_start * 0.15 + plot_x_end * 0.85, plot_y_start * 0.65 + plot_y_end * 0.35,
                vmin_str, ha="right", va="top", fontsize=8)
        ax.text(plox_x_start * 0.15 + plot_x_end * 0.85, plot_y_start * 0.95 + plot_y_end * 0.05,
                vmax_str, ha="right", va="bottom", fontsize=8)
        ax.set_xlim(plox_x_start, plot_x_end)
        ax.set_ylim(plot_y_start, plot_y_end)

        ax.set_title(lib_figures.style_variable_name(feature))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide the last axes
    for i in range(len(features), len(axes)):
        axes[i].axis("off")

    plt.show()


if __name__ == "__main__":
    main()
