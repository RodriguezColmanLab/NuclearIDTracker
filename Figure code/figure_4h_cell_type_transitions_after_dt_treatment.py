import numpy
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.patches import Arc

import figure_3_sup_cell_type_transition_matrix as cell_type_transition_panel
import lib_figures

_AVERAGING_WINDOW_WIDTH_H = 5
_DATASET_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATASET_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"


PANETH_CELL = cell_type_transition_panel.CellTypeInMatrix(0, 0, includes_paneth=True)
STEM_TO_TA_CELL = cell_type_transition_panel.CellTypeInMatrix(0.5, 1.0, includes_paneth=False)
ENTEROCYTE_CELL = cell_type_transition_panel.CellTypeInMatrix(0.0, 0.5, includes_paneth=False)


def main():
    # Collect data
    transition_matrix_control = cell_type_transition_panel.calculate_transition_matrix(_DATASET_FILE_CONTROL)
    transition_matrix_regeneration = cell_type_transition_panel.calculate_transition_matrix(_DATASET_FILE_REGENERATION)

    _show_arrows_figure(transition_matrix_control, transition_matrix_regeneration)


def _show_arrows_figure(control_summed_data: cell_type_transition_panel.CellTypeTransitionMatrix,
                        regeneration_summed_data: cell_type_transition_panel.CellTypeTransitionMatrix):
    figure = lib_figures.new_figure(size=(5, 4))
    ax_control, ax_regen = figure.subplots(ncols=2, nrows=1)
    ax_control.set_title("Control")
    _plot_arrows(ax_control, control_summed_data)
    ax_regen.set_title("Regeneration")
    _plot_arrows(ax_regen, regeneration_summed_data)
    figure.tight_layout()
    plt.show()


def _plot_arrows(ax: Axes, transition_data: cell_type_transition_panel.CellTypeTransitionMatrix):
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")

    # Add dots for the cell types
    cell_types = [STEM_TO_TA_CELL, ENTEROCYTE_CELL, PANETH_CELL]
    x_positions = [-1, .5, .5]
    y_positions = [0, 1.7, -1.7]
    ax.scatter(x_positions, y_positions, s=1000,
               c=[cell_type.color() for cell_type in cell_types], zorder=2)

    transition_counts = [transition_data.get_transition_count(STEM_TO_TA_CELL, to_cell_type) for to_cell_type in cell_types]
    print(transition_counts)
    x_cell_type_from = x_positions[0]
    y_cell_type_from = y_positions[0]
    for i, count in enumerate(transition_counts):
        if i == 0:
            # Stem cell to stem cell transitions
            x = x_cell_type_from - 0.2
            y = y_cell_type_from + 0.75
            _circular_arrow(ax, 0.5, x, y, 310, 270, color="black")
            ax.text(x - 0.3, y, f"{count / sum(transition_counts) * 100:.1f}%",
                    ha="right", va="center", zorder=3)
            continue

        x_cell_type_to = x_positions[i]
        y_cell_type_to = y_positions[i]

        # Draw arrow from position 33% along the line to 66% along the line
        x_start = x_cell_type_from * 0.66 + x_cell_type_to * 0.33
        y_start = y_cell_type_from * 0.66 + y_cell_type_to * 0.33
        x_end = x_cell_type_from * 0.33 + x_cell_type_to * 0.66
        y_end = y_cell_type_from * 0.33 + y_cell_type_to * 0.66

        ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, head_width=0.1, head_length=0.1, fc='black',
                 ec='black', zorder=1)

        # Add percentage to the arrow
        ax.text((x_start + x_end) / 2, (y_start + y_end) / 2, f"{count / sum(transition_counts) * 100:.1f}%",
                ha="center", va="center", zorder=3)


def _circular_arrow(ax: Axes, diameter: float, cent_x: float, cent_y: float, start_angle: float, angle: float, **kwargs):
    startarrow = kwargs.pop("startarrow", False)
    endarrow = kwargs.pop("endarrow", False)

    arc = Arc([cent_x, cent_y], diameter, diameter, angle=start_angle,
              theta1=numpy.rad2deg(kwargs.get("head_length", 1.5 * 3 * .001)) if startarrow else 0,
              theta2=angle - (numpy.rad2deg(kwargs.get("head_length", 1.5 * 3 * .001)) if endarrow else 0),
              linestyle="-", color=kwargs.get("color", "black"))
    ax.add_patch(arc)

    if startarrow:
        startX = diameter / 2 * numpy.cos(numpy.radians(start_angle))
        startY = diameter / 2 * numpy.sin(numpy.radians(start_angle))
        startDX = +.000001 * diameter / 2 * numpy.sin(
            numpy.radians(start_angle) + kwargs.get("head_length", 1.5 * 3 * .001))
        startDY = -.000001 * diameter / 2 * numpy.cos(
            numpy.radians(start_angle) + kwargs.get("head_length", 1.5 * 3 * .001))
        ax.arrow(startX - startDX, startY - startDY, startDX, startDY, **kwargs)

    if endarrow:
        endX = diameter / 2 * numpy.cos(numpy.radians(start_angle + angle))
        endY = diameter / 2 * numpy.sin(numpy.radians(start_angle + angle))
        endDX = -.000001 * diameter / 2 * numpy.sin(
            numpy.radians(start_angle + angle) - kwargs.get("head_length", 1.5 * 3 * .001))
        endDY = +.000001 * diameter / 2 * numpy.cos(
            numpy.radians(start_angle + angle) - kwargs.get("head_length", 1.5 * 3 * .001))
        ax.arrow(endX - endDX, endY - endDY, endDX, endDY, **kwargs)


if __name__ == "__main__":
    main()
