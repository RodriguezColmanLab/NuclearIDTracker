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
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the LDA
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    plot_coords = lda.fit(adata.X, adata.obs["cell_type_training"]).transform(adata.X)

    # Plot the LDA
    figure = lib_figures.new_figure(size=(4.5, 3.5))
    ax: Axes = figure.gca()
    used_cell_types = adata.obs["cell_type_training"].array.categories
    colors = [lib_figures.CELL_TYPE_PALETTE[cell_type] for cell_type in adata.obs["cell_type_training"]]
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1], s=15, lw=0, c=colors)

    ax.set_title("Linear Discriminant Analysis")
    # Add legend for all the cell types
    for cell_type in used_cell_types:
        ax.scatter([], [], label=lib_figures.style_cell_type_name(cell_type), s=15, lw=0,
                   color=lib_figures.CELL_TYPE_PALETTE[cell_type])
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Cell type")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
