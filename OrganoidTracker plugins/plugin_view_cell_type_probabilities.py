from functools import partial
from typing import Any, Dict

import matplotlib
from matplotlib.colors import Colormap

from organoid_tracker.core.position import Position
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> dict[str, Any]:
    return {
        "View//Types-Cell type probabilities...": lambda: _show_probabilities(window)
    }


def _show_probabilities(window: Window):
    activate(_CellTypeProbabilitiesVisualizer(window))



class _CellTypeProbabilitiesVisualizer(ExitableImageVisualizer):
    """Shows the probabilities of a single cell type. Use the Cell type menu to select which cell type to visualize."""

    _cell_type: str | None = None
    _cmap: Colormap = matplotlib.colormaps.get_cmap("magma")

    def get_extra_menu_options(self) -> Dict[str, Any]:
        cell_types = self._experiment.global_data.get_data("ct_probabilities")
        if not isinstance(cell_types, list):
            return super().get_extra_menu_options()

        menu_options = super().get_extra_menu_options()
        for cell_type in cell_types:
            menu_options[f"Cell type//Types-{cell_type}"] = partial(self._set_cell_type, cell_type)
        return menu_options

    def _set_cell_type(self, cell_type: str):
        self._cell_type = cell_type
        self.draw_view()
        self.update_status(f"Now showing probabilities for cell type {cell_type}.")

    def _get_figure_title(self) -> str:
        return super()._get_figure_title() + f"\nCell type: {self._cell_type}"

    def _draw_missing_type(self, position: Position):
        self._ax.scatter([position.x], [position.y], color=(0, 0, 0, 0), s=300, lw=2, edgecolors="#eeeeee", linestyle="dashed")

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if self._cell_type is None:
            return True
        cell_types = self._experiment.global_data.get_data("ct_probabilities")
        if not isinstance(cell_types, list) or not self._cell_type in cell_types:
            return True

        cell_type_index = cell_types.index(self._cell_type)
        probabilities = self._experiment.positions.get_position_data(position, "ct_probabilities")
        if not isinstance(probabilities, list) or len(probabilities) <= cell_type_index:
            self._draw_missing_type(position)
            return False
        probability = probabilities[cell_type_index]
        if not isinstance(probability, float):
            self._draw_missing_type(position)
            return False

        # Draw a circle with the color corresponding to the probability of being of the given cell type
        self._ax.scatter([position.x], [position.y], color=self._cmap(probability), s=300, lw=0)
        return False