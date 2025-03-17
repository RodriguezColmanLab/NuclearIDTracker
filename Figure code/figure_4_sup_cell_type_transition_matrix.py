from matplotlib import pyplot as plt

import figure_3_sup_cell_type_transition_matrix as cell_type_transition_panel
import lib_figures

_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


def main():
    transition_matrix = cell_type_transition_panel.calculate_transition_matrix(_DATA_FILE_REGENERATION)

    figure = lib_figures.new_figure(size=(5, 3.5))
    ax_matrix, ax_scalebar = figure.subplots(nrows=1, ncols=2, width_ratios=[1, 0.03])

    scaleable = cell_type_transition_panel.plot_transition_matrix(ax_matrix, transition_matrix)
    figure.colorbar(scaleable, cax=ax_scalebar).set_label("Cell count")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
