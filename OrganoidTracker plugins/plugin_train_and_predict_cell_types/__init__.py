from typing import Dict, Any, List

from organoid_tracker.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "View//Types-Cell type calculations (linear model only)...": lambda: _view_cell_type_parameters(window),
        "Tools//Process-Cell types//3. Predict cell types...": lambda: _create_prediction_script(window)
    }


def get_commands():
    return {
        "types_predict": _predict_types,
    }


def _view_cell_type_parameters(window: Window):
    from . import inspect_type_calculation
    inspect_type_calculation.view_cell_type_parameters(window)


def _create_prediction_script(window: Window):
    from . import predict_on_organoidtracker_data
    predict_on_organoidtracker_data.create_prediction_script(window)


def _predict_types(args: List[str]):
    from . import predict_on_organoidtracker_data
    predict_on_organoidtracker_data.run_predictions()
    return 0
