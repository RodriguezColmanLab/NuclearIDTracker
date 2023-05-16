"""Calculates the cell type transitin matrix for the cell types stem, enterocyte, Paneth, other secretory."""
from organoid_tracker.core import UserError
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window):
    return {
        "Tools//Process-Cell types//Calculate cell type transition matrix...": lambda: _calculate_transition_matrix(window)
    }


def _calculate_transition_matrix(window: Window):
    raise UserError("Not yet implemented", "This function is not yet available.")
