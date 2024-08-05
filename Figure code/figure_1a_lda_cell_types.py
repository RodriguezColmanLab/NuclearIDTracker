import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
import sklearn.discriminant_analysis
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures

INPUT_FILE = "../../Data/all_data.h5ad"


def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(INPUT_FILE)
    adata = lib_figures.standard_preprocess(adata)

    # Remove cells that we cannot train on
    #adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the LDA
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    plot_coords = lda.fit(adata.X, adata.obs["cell_type_training"]).transform(adata.X)

    # Plot the LDA
    figure = lib_figures.new_figure(size=(4.5, 4))
    ax: Axes = figure.gca()
    used_cell_types = adata.obs["cell_type_training"].array.categories
    for cell_type in used_cell_types:
        depth = 0 if cell_type == "NONE" else 3
        mask = adata.obs["cell_type_training"] == cell_type
        ax.scatter(plot_coords[mask, 0], plot_coords[mask, 1],
                   s=10, lw=0, zorder=depth,
                   color=lib_figures.CELL_TYPE_PALETTE[cell_type], label=lib_figures.style_cell_type_name(cell_type))

    ax.set_title("Linear Discriminant Analysis")
    ax.legend(
        loc='center left', bbox_to_anchor=(1, 0.5)
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    main()
