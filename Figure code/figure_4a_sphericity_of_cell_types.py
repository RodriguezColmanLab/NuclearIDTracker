import numpy
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
import scipy
from matplotlib import pyplot as plt

import lib_figures
import lib_data

INPUT_FILE = "../../Data/all_data.h5ad"

def main():
    adata = scanpy.read_h5ad(INPUT_FILE)

    # Standard preprocessing, but without scaling (also undo the automatic scaling of some features)
    adata = lib_figures.standard_preprocess(adata, scale=False, log1p=False)
    for feature_index, feature_name in enumerate(adata.var_names):
        values = adata.X[:, feature_index]
        if lib_data.should_be_exponential(feature_name):
            adata.X[:, feature_index] = numpy.log(values)

    # De-duplicate the cells by only taking the ones with the highest time point per organoid
    adata = lib_data.deduplicate_cells(adata)

    # Plot the sphericity of the cell types as a violin plot
    volume_index = adata.var_names.get_loc("volume_um3")
    surface_index = adata.var_names.get_loc("surface_um2")
    volumes = adata.X[:, volume_index]
    surfaces = adata.X[:, surface_index]
    sphericities = numpy.pi ** (1 / 3) * (6 * volumes) ** (2 / 3) / surfaces

    cell_types = ["MATURE_GOBLET", "PANETH", "STEM", "ENTEROCYTE"]
    all_cell_types_sphericities = []
    all_cell_types_names = []
    all_cell_types_colors = []
    for cell_type in cell_types:
        cell_type_indices = adata.obs["cell_type_training"] == cell_type
        cell_type_sphericities = sphericities[cell_type_indices]

        all_cell_types_sphericities.append(cell_type_sphericities)
        all_cell_types_names.append(lib_figures.style_cell_type_name(cell_type))
        all_cell_types_colors.append(lib_figures.CELL_TYPE_PALETTE[cell_type])




    figure = lib_figures.new_figure(size=(2.5, 3))
    ax = figure.gca()
    violins = ax.violinplot(all_cell_types_sphericities, showmeans=True, showextrema=False, widths=0.8)
    for violin, color in zip(violins["bodies"], all_cell_types_colors):
        violin.set_facecolor(color)
        violin.set_alpha(1)
    violins["cmeans"].set_color("black")
    ax.set_xticks(numpy.arange(1, len(all_cell_types_names) + 1))
    ax.set_xticklabels(all_cell_types_names)
    ax.set_ylabel("Sphericity")
    ax.set_title(" ")  # Just to make some room above the plot

    # Add N=... to the plot
    for i in range(len(cell_types)):
        ax.text(i + 1, 0.90, f"{len(all_cell_types_sphericities[i])}", ha="center", va="center")

    # Do a t-test
    stem_index = cell_types.index("STEM")
    enterocyte_index = cell_types.index("ENTEROCYTE")
    result = scipy.stats.ttest_ind(all_cell_types_sphericities[stem_index],
                                   all_cell_types_sphericities[enterocyte_index])
    ax.plot([stem_index + 1, enterocyte_index + 1], [0.875, 0.875], color="black", lw=1)
    ax.text((stem_index + enterocyte_index + 2) / 2, 0.88, f"p={result.pvalue:.4f}", ha="center", va="center")

    figure.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
