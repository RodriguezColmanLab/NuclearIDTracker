import pandas
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
import sklearn.discriminant_analysis
from matplotlib import pyplot as plt

import lib_figures

INPUT_FILE = "../../Data/all_data.h5ad"
OUTPUT_FILE = r"S:\_RodriguezColman group share\Group members\Maria\NuclearIDTracker\Table S1.xlsx"

def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(INPUT_FILE)
    adata = lib_figures.standard_preprocess(adata)  # So only filtering

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the LDA
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(adata.X, adata.obs["cell_type_training"])

    # Print the scalings table as a formatted figure
    scalings_table = pandas.DataFrame(lda.scalings_,
                                      index=[lib_figures.style_variable_name(var_name) for var_name in adata.var_names],
                                      columns=[f"LDA {i + 1}" for i in range(lda.scalings_.shape[1])])
    # Sort table by first column
    scalings_table.sort_values(by="LDA 1", ascending=False, inplace=True)

    figure = lib_figures.new_figure(size=(3, 4))
    ax = figure.gca()
    for col_number in range(scalings_table.shape[1]):
        for row_number in range(scalings_table.shape[0]):
            value = scalings_table.iloc[row_number, col_number]
            ax.text(col_number, row_number, f"{value:.2f}", ha="center", va="center")

    ax.set_xlim(-1, scalings_table.shape[1])
    ax.set_ylim(-0.5, scalings_table.shape[0] + 0.3)

    ax.set_yticks(list(range(scalings_table.shape[0])))
    ax.set_yticklabels(list(scalings_table.index), rotation=0, ha="right", va="center")

    ax.xaxis.set_ticks_position('top')  # the rest is the same
    ax.set_xticks(list(range(scalings_table.shape[1])))
    ax.set_xticklabels([f"LDA {i + 1}" for i in range(scalings_table.shape[1])], ha="center", va="bottom")

    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.tight_layout()
    plt.show()

    # To save as a CSV/TSV, use this
    scalings_table.to_excel(OUTPUT_FILE)


if __name__ == "__main__":
    main()
