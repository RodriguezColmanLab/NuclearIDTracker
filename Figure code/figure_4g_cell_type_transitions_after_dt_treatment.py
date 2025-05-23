import matplotlib.colors
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
STEM_CELL = cell_type_transition_panel.CellTypeInMatrix(0.55, 1.0, includes_paneth=False)
TA_CELL = cell_type_transition_panel.CellTypeInMatrix(0.4, 0.55, includes_paneth=False)
ENTEROCYTE_CELL = cell_type_transition_panel.CellTypeInMatrix(0.0, 0.4, includes_paneth=False)


def main():
    # Collect data
    transition_matrix_control = cell_type_transition_panel.calculate_transition_matrix(_DATASET_FILE_CONTROL)
    transition_matrix_regeneration = cell_type_transition_panel.calculate_transition_matrix(_DATASET_FILE_REGENERATION)

    _show_arrows_figure(transition_matrix_control, transition_matrix_regeneration)


def _show_arrows_figure(control_summed_data: cell_type_transition_panel.CellTypeTransitionMatrix,
                        regeneration_summed_data: cell_type_transition_panel.CellTypeTransitionMatrix):
    figure = lib_figures.new_figure(size=(5, 4))
    ax_control, ax_regen = figure.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
    ax_control.set_title("Control")
    _plot_arrows(ax_control, control_summed_data)
    ax_regen.set_title("Regeneration")
    _plot_arrows(ax_regen, regeneration_summed_data)
    figure.tight_layout()
    plt.show()


def _plot_arrows(ax: Axes, transition_data: cell_type_transition_panel.CellTypeTransitionMatrix):
    ax.set_xlim(-4.5, 3)
    ax.set_ylim(-2.5, 5)
    ax.set_aspect("equal")

    # Define what we want to plot
    cell_types = [STEM_CELL, TA_CELL, ENTEROCYTE_CELL, PANETH_CELL]
    cell_type_transitions = [
        (STEM_CELL, STEM_CELL),
        (STEM_CELL, TA_CELL),
        (TA_CELL, STEM_CELL),
        (TA_CELL, ENTEROCYTE_CELL),
        (TA_CELL, TA_CELL),
        (TA_CELL, PANETH_CELL),
        (ENTEROCYTE_CELL, TA_CELL),
        (ENTEROCYTE_CELL, ENTEROCYTE_CELL),
        (ENTEROCYTE_CELL, PANETH_CELL),
    ]
    positions_xy = {
        STEM_CELL: (-3, 0),
        TA_CELL: (-1, 0),
        ENTEROCYTE_CELL: (0.5, 1.7),
        PANETH_CELL: (0.5, -1.7)
    }

    # Add dots for the cell types
    ax.scatter([positions_xy[cell_type][0] for cell_type in cell_types],
               [positions_xy[cell_type][1] for cell_type in cell_types],
               s=500, c=[cell_type.color() for cell_type in cell_types], zorder=2)

    # Add counts for the cell types
    end_counts = [sum(transition_data.get_transition_count(cell_type_from, cell_type_to) for cell_type_from in cell_types) for cell_type_to in cell_types]
    for i, cell_type in enumerate(cell_types):
        x = positions_xy[cell_type][0]
        y = positions_xy[cell_type][1]

        cell_type_hue = matplotlib.colors.rgb_to_hsv(cell_type.color())[0]
        print(cell_type, cell_type_hue)
        text_color = "black" if 0.1 < cell_type_hue < 0.6 else "white"
        ax.text(x, y, str(int(end_counts[i])), ha="center", va="center", fontsize=7, zorder=3, color=text_color)


    # Add the transitions
    for cell_type_from, cell_type_to in cell_type_transitions:
        transition_count = transition_data.get_transition_count(cell_type_from, cell_type_to)
        total_count = sum(transition_data.get_transition_count(cell_type_from, other_cell_type) for other_cell_type in cell_types)
        if total_count == 0 or transition_count == 0:
            continue  # No transition data for this cell type

        x_cell_type_from = positions_xy[cell_type_from][0]
        y_cell_type_from = positions_xy[cell_type_from][1]

        if cell_type_from == cell_type_to:
            # Self-transition
            x = x_cell_type_from - 0.2
            y = y_cell_type_from + 0.75
            _circular_arrow(ax, 0.5, x, y, 310, 270, color="black")
            ax.text(x - 0.3, y, f"{transition_count / total_count * 100:.0f}%",
                    ha="right", va="center", zorder=3)
            continue

        x_cell_type_to = positions_xy[cell_type_to][0]
        y_cell_type_to = positions_xy[cell_type_to][1]

        # Draw arrow from position 33% along the line to 66% along the line
        x_start = x_cell_type_from * 0.66 + x_cell_type_to * 0.33
        y_start = y_cell_type_from * 0.66 + y_cell_type_to * 0.33
        x_end = x_cell_type_from * 0.33 + x_cell_type_to * 0.66
        y_end = y_cell_type_from * 0.33 + y_cell_type_to * 0.66

        # Move arrow slightly to
        text_alignment = "right"
        x_text = (x_start + x_end) / 2
        y_text = (y_start + y_end) / 2
        if y_start == y_end:
            # Horizontal arrow, adjust y to avoid overlap
            text_alignment = "center"
            if x_start < x_end:
                y_start -= 0.2
                y_end -= 0.2
                y_text -= 0.4
            else:
                y_text += 0.1
        elif y_start < y_end:
            # Vertical or diagonal arrow, adjust x to avoid overlap
            x_start += 0.2
            x_end += 0.2
            x_text += 0.4
            text_alignment = "left"

        ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, head_width=0.1, head_length=0.1, fc='black',
                 ec='black', zorder=1)

        # Add percentage to the arrow
        ax.text(x_text, y_text, f"{transition_count / total_count * 100:.0f}%",
                ha=text_alignment, va="center", zorder=3)


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
