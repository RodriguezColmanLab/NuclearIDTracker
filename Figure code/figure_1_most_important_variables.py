from typing import List

import numpy
import scanpy.tools
import scanpy.plotting
from matplotlib import pyplot as plt
import matplotlib.colors
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from numpy import ndarray

import figure_lib

INPUT_FILE = "../../Data/all_data.h5ad"
NUMBER_OF_GENES = 6
COLORMAP = LinearSegmentedColormap.from_list("custom", ["#12662E", "#136D31", "#157536", "#17803C", "#48AE60", "#F6FCF4"])

def main():
    adata = scanpy.read_h5ad(INPUT_FILE)

    # Standard preprocessing
    adata = figure_lib.standard_preprocess(adata)

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Nicer names
    adata.var_names = [figure_lib.style_variable_name(var_name)
                       .replace(" ", "\n")
                       .replace("\n(", " (")
                       .replace("minor\naxis", "minor axis")
                       .replace("major\naxis", "major axis")
                       for var_name in adata.var_names]
    adata.obs["cell_type_training"] = [figure_lib.style_cell_type_name(cell_type) for cell_type in adata.obs["cell_type_training"]]

    scanpy.tools.rank_genes_groups(adata, 'cell_type_training', method='wilcoxon')
    #scanpy.plotting.rank_genes_groups(adata, n_genes=25, sharey=False)

    cell_types = list(adata.obs["cell_type_training"].array.categories)

    name_table = numpy.full(shape=(NUMBER_OF_GENES, len(cell_types)), fill_value="---", dtype=object)
    score_table = numpy.zeros(shape=name_table.shape, dtype=numpy.float64)

    for i, cell_type in enumerate(cell_types):
        gene_names = adata.uns["rank_genes_groups"]['names'][cell_type][:NUMBER_OF_GENES]
        scores = adata.uns["rank_genes_groups"]['pvals_adj'][cell_type][:NUMBER_OF_GENES]
        name_table[:, i] = gene_names
        score_table[:, i] = scores


    plot(figure_lib.new_figure(size=(6, 2.6)), cell_types, name_table, score_table)
    plt.show()



def plot(figure: Figure, cell_types: List[str], name_table: ndarray, score_table: ndarray):
    ax = figure.gca()
    image = ax.imshow(score_table, aspect=1/2, cmap=COLORMAP,
                      norm=matplotlib.colors.LogNorm(vmin=score_table.min(), vmax=0.1))

    ax.set_yticks([])
    ax.set_xticks(list(range(len(cell_types))))
    ax.set_xticklabels(cell_types)
    ax.xaxis.tick_top()

    for i in range(name_table.shape[0]):
        for j in range(name_table.shape[1]):
            color = "white" if score_table[i, j] < 1e-11 else "black"
            text = name_table[i, j]
            if score_table[i, j] > 0.05:
                striked_out = ""
                for character in text:
                    striked_out += character + '\u0336'
                text = striked_out
            ax.text(j, i, text, horizontalalignment="center", verticalalignment="center", color=color)

    colorbar: Colorbar = figure.colorbar(image)
    colorbar.ax.set_ylabel("p-value (FDR-corrected)")


if __name__ == "__main__":
    main()
