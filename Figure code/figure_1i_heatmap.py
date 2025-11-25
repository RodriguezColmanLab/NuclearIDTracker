import numpy
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
import scipy
from matplotlib import pyplot as plt

import lib_figures

INPUT_FILE = "../../Data/all_data.h5ad"

def main():
    adata = scanpy.read_h5ad(INPUT_FILE)

    # Standard preprocessing
    adata = lib_figures.standard_preprocess(adata)

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Nicer names
    adata.var_names = [lib_figures.style_variable_name(var_name) for var_name in adata.var_names]
    adata.obs["cell_type_training"] = [lib_figures.style_cell_type_name(cell_type) for cell_type in adata.obs["cell_type_training"]]

    # Make the adata object have random ordering of cells
    adata = adata[numpy.random.permutation(adata.obs_names)]

    # Plot!
    _ = lib_figures.new_figure()
    clustering_variables_result = scipy.cluster.hierarchy.linkage(adata.X.T, method="average")
    reordering = scipy.cluster.hierarchy.leaves_list(clustering_variables_result)
    var_names = numpy.array(adata.var_names)
    var_names = var_names[reordering]

    scanpy.plotting.heatmap(adata,
                            var_names=var_names,
                            groupby="cell_type_training",
                            cmap="bwr",
                            swap_axes=True,
                            vmin=-3,
                            vmax=3)

    # Print how many cells and organoids we have
    n_cells = adata.n_obs
    n_organoids = adata.obs["organoid"].nunique()
    print(f"Number of cells: {n_cells}")
    print(f"Number of organoids: {n_organoids}")

    # Calculate p-value for each variable per organoid; is there a significant difference between cell types?
    adata_organoids = scanpy.get.aggregate(adata, by=["organoid", "cell_type_training"], func="mean")
    p_values = list()
    for var_name in var_names:
        groups = []
        for cell_type in adata_organoids.obs["cell_type_training"].cat.categories:
            mask = adata_organoids.obs["cell_type_training"] == cell_type
            groups.append(adata_organoids[mask, var_name].layers["mean"].toarray().flatten())
        stat, p_value = scipy.stats.kruskal(*groups)
        p_values.append(p_value)
    p_values = numpy.array(p_values)

    # Correct for multiple testing using FDR
    p_values = scipy.stats.false_discovery_control(p_values, method="bh")

    bar_colors = ["#0984e3" if p_value < 0.05 else "#dfe6e9" for p_value in p_values]

    # Make a bar plot of the p-values
    figure = lib_figures.new_figure(size=(1.2, 3))
    ax = figure.gca()
    ax.barh(range(len(var_names)), width=p_values - 1, left=1, color=bar_colors)
    ax.set_yticks(range(len(var_names)))
    ax.set_yticklabels(var_names)
    ax.set_xlabel("Organoid-aggregated FDR-corrected\nKruskal-Wallis p-value")
    ax.axvline(0.05, color="#0984e3")
    ax.text(0.045, 0, "p=0.05", color="#0984e3", ha="left", va="top")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(len(var_names) - 0.5, -0.5)
    plt.show()

    # To visualize the data behind the p-values
    # scanpy.plotting.heatmap(adata_organoids,
    #                         var_names=var_names,
    #                         groupby="cell_type_training",
    #                         cmap="bwr",
    #                         swap_axes=True,
    #                         layer="mean",
    #                         vmin=-3,
    #                         vmax=3)


if __name__ == "__main__":
    main()
