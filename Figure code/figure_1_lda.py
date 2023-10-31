import math

import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
import sklearn.discriminant_analysis
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

import lib_figures

INPUT_FILE = "../../Data/all_data.h5ad"


def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(INPUT_FILE)
    adata = lib_figures.standard_preprocess(adata, filter=False)

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the PCA
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    plot_coords = lda.fit(adata.X, adata.obs["cell_type_training"]).transform(adata.X)

    # Plot the PCA
    figure = lib_figures.new_figure(size=(3.5, 2.5))
    ax: Axes = figure.gca()
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1],
               alpha=0.8, s=12, lw=0,
               color=[lib_figures.CELL_TYPE_PALETTE[adata.obs["cell_type_training"][i]] for i in
                      range(len(adata.obs["cell_type_training"]))])
    used_cell_types = adata.obs["cell_type_training"].array.categories

    ax.set_title("Linear Discriminant Analysis")
    ax.legend(handles=[
        Line2D([0], [0], marker='o', alpha=0.8,
               color=lib_figures.CELL_TYPE_PALETTE[cell_type],
               label=lib_figures.style_cell_type_name(cell_type),
               markersize=math.sqrt(12), lw=0)
        for cell_type in used_cell_types],
        loc='center left', bbox_to_anchor=(1, 0.5)
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    main()
