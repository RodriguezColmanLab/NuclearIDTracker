"""Calculates the cell type transitin matrix for the cell types stem, enterocyte, Paneth, other secretory."""
from collections import defaultdict
from typing import Dict, Tuple

from organoid_tracker.core import UserError
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window):
    return {
        "Tools//Process-Cell types//Calculate cell type transition matrix...": lambda: _calculate_transition_matrix(window)
    }


class _TransitionMatrix:

    _transition_matrix: Dict[Tuple[str, str], int]

    def __init__(self):
        self._transition_matrix = defaultdict(lambda: 0)

    def record_transition(self, from_type: str, to_type: str):
        self._transition_matrix[(from_type, to_type)] += 1


def _calculate_transition_matrix(window: Window):
    transition_matrix = _TransitionMatrix()
    for experiment in window.get_active_experiments():
        for track in experiment.links.find_ending_tracks():
            ...
