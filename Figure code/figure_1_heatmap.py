import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools

import figure_lib

INPUT_FILE = "../../Data/all_data.h5ad"

def main():
    adata = scanpy.read_h5ad(INPUT_FILE)

    # Standard preprocessing
    adata = figure_lib.standard_preprocess(adata)

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Nicer names
    adata.var_names = [figure_lib.style_variable_name(var_name) for var_name in adata.var_names]
    adata.obs["cell_type_training"] = [cell_type.lower() for cell_type in adata.obs["cell_type_training"]]

    # Plot!
    _ = figure_lib.new_figure()
    scanpy.plotting.heatmap(adata,
                            var_names=adata.var_names,
                            groupby="cell_type_training",
                            cmap="PiYG_r",
                            swap_axes=True,
                            vmin=-2,
                            vmax=2)


if __name__ == "__main__":
    main()
