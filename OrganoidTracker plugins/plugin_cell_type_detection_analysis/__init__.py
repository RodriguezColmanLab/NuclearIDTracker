from typing import List, Dict, Callable, Any

from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import activate


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export positions//Metadata-CSV, with cell type probabilities...": lambda: _export_probabilities(window),
        "Edit//Types-Predicted types//Average the cell types": lambda: _average_cell_types(window),
        "Edit//Types-Predicted types//Viterbi the cell types...": lambda: _viterbi_cell_types(window),
        "Edit//Types-Predicted types//Back to original cell types": lambda: _unaverage_cell_types(window),
        "View//Analyze-Cell type probabilities...": lambda: _show_probabilities(window)
    }


def _export_probabilities(window: Window):
    from ._probabilities_exporter import export_probabilities_to_csv
    export_probabilities_to_csv(window)


def _average_cell_types(window: Window):
    from ._cell_types_averager import average_cell_types
    average_cell_types(window)


def _unaverage_cell_types(window: Window):
    from ._cell_types_averager import unaverage_cell_types
    unaverage_cell_types(window)


def _viterbi_cell_types(window: Window):
    from ._viterbi_applier import apply_viterbi
    from ._viterbi import TRANSITION_FILE
    file = dialog.prompt_load_file("Cell type transition file", [("Transition file", TRANSITION_FILE)])
    if file is None:
        return
    for experiment in window.get_active_experiments():
        apply_viterbi(experiment, file)
    window.redraw_data()
    window.set_status("Applied the Viterbi algorithm to the cell types.")


def _show_probabilities(window: Window):
    from ._cell_type_visualizer import CellTypeVisualizer
    activate(CellTypeVisualizer(window))
