import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
from matplotlib import pyplot as plt

import figure_lib

INPUT_FILE = "../../Data/all_data.h5ad"


def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(INPUT_FILE)
    adata = figure_lib.standard_preprocess(adata)

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the PCA
    scanpy.tools.pca(adata, svd_solver='arpack')

    # Plot the PCA
    figure = figure_lib.new_figure(size=(3.5,2.5))
    ax = figure.gca()
    scanpy.plotting.pca(adata, ax=ax, annotate_var_explained=True, color="cell_type_training",
                        palette=figure_lib.CELL_TYPE_PALETTE, show=False, s=35, alpha=0.5)
    ax.set_title("Principal Component Analysis")
    plt.show()


if __name__ == "__main__":
    main()
