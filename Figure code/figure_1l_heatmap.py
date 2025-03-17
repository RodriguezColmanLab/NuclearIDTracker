import numpy
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
import scipy

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

    scanpy.plotting.heatmap(adata,
                            var_names=var_names[reordering],
                            groupby="cell_type_training",
                            cmap="bwr",
                            swap_axes=True,
                            vmin=-3,
                            vmax=3)


if __name__ == "__main__":
    main()
