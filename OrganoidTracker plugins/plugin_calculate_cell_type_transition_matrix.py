"""Calculates the cell type transitin matrix for the cell types stem, enterocyte, Paneth, other secretory."""
from collections import defaultdict
from typing import Dict, Tuple, Optional, List, Union

from organoid_tracker.core import UserError
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import position_markers


def get_menu_items(window: Window):
    return {
        "Tools//Process-Cell types//Transitions-Calculate cell type transition matrix...": lambda: _calculate_transition_matrix(window)
    }


class _TransitionMatrix:

    _transition_matrix: Dict[Tuple[str, str], int]

    def __init__(self):
        self._transition_matrix = defaultdict(lambda: 0)

    def record_transition(self, from_type: str, to_type: str, weight: float = 1):
        self._transition_matrix[(from_type, to_type)] += weight

    def serialize(self) -> Dict[str, Dict[str, int]]:
        """Serializes this object to a json-compatible dictionary."""
        return_dict = dict()
        for (from_type, to_type), count in self._transition_matrix.items():
            if from_type not in return_dict:
                return_dict[from_type] = dict()
            return_dict[from_type][to_type] = count
        return return_dict


def _convert_cell_type(position_type: Optional[str]) -> Union[Tuple, Tuple[str], Tuple[str, str]]:
    """Conversion for Viterbi algorithm, to estimate transition rates. Assumes some precursor cells to be of their
    final type.

    Zero, one or two cell types can be returned. If two are returned, then there are two possible alternatives. If two
    subsequent time points return two cell types, then add a transition from type 1 to type 1, and type 2 to type 2.
    If one time point has two types, and another 1, then add transitions for both type 1 and 2 to the single type.
    """
    if position_type is None:
        return ()
    if position_type == "ENTEROCYTE":
        return "ENTEROCYTE",
    if position_type == "ABSORPTIVE_PRECURSOR":
        return "STEM", "ENTEROCYTE"
    if position_type == "UNLABELED":
        return "STEM", "UNLABELED"
    if position_type in {"ENTEROENDOCRINE", "GOBLET", "TUFT", "SECRETORY"}:
        # Seeing the difference between these types is hard for the network
        return "OTHER_SECRETORY",
    if position_type == "SECRETIVE_PRECURSOR":
        return "STEM", "OTHER_SECRETORY"
    if position_type in {"PANETH", "WGA_PLUS"}:
        return "PANETH",
    if position_type in {"STEM", "STEM_PUTATIVE"}:
        return "STEM",
    if position_type == "MATURE_GOBLET":
        return "MATURE_GOBLET",
    return ()


def _calculate_transition_matrix(window: Window):
    transition_matrix = _TransitionMatrix()
    for experiment in window.get_active_experiments():
        position_data = experiment.position_data

        for position_from, position_to in experiment.links.find_all_links():
            cell_types_from = _convert_cell_type(position_markers.get_position_type(position_data, position_from))
            cell_types_to = _convert_cell_type(position_markers.get_position_type(position_data, position_to))

            if len(cell_types_from) == 0 and len(cell_types_to) > 0:
                cell_types_from = ["STEM", "UNLABELED"]  # Assume all cells came from stem cells

            if len(cell_types_from) == len(cell_types_to):
                # Kept same number of possibilities, add transitions from respective cell types
                for i in range(len(cell_types_from)):
                    transition_matrix.record_transition(cell_types_from[i], cell_types_to[i], weight=1/len(cell_types_from))
            else:
                # Changed number of possibilities, allow all possible transitions
                transitions = len(cell_types_from) * len(cell_types_to)
                for cell_type_from in cell_types_from:
                    for cell_type_to in cell_types_to:
                        transition_matrix.record_transition(cell_type_from, cell_type_to, weight=1/transitions)

    print(transition_matrix.serialize())