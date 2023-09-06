import math
import pickle
from typing import Dict, Any, Optional, NamedTuple, List

import numpy
import scipy
from matplotlib.backend_bases import MouseEvent
from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.text_popup.text_popup import RichTextPopup
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "View//Types-Cell type calculations (linear model only)...": lambda: _view_cell_type_parameters(window)
    }


class _ModelParameters(NamedTuple):
    """Not the original class, but uses Python's duck typing so that we know which fields exist."""
    cell_type_mapping: List[str]
    input_mapping: List[str]
    scaler: StandardScaler
    regressor: LogisticRegression


def _get_data_array(position_data: PositionData, position: Position, input_names: List[str]) -> Optional[ndarray]:
    array = numpy.empty(len(input_names), dtype=numpy.float32)
    for i, name in enumerate(input_names):
        value = None
        if name == "sphericity":
            # Special case, we need to calculate
            volume = position_data.get_position_data(position, "volume_um3")
            surface = position_data.get_position_data(position, "surface_um2")
            if volume is not None and surface is not None:
                value = math.pi ** (1/3) * (6 * volume) ** (2/3) / surface
        else:
            # Otherwise, just look up
            value = position_data.get_position_data(position, name)

        if value is None or value == 0:
            return None  # Abort, a value is missing

        if name in {"neighbor_distance_variation", "solidity", "sphericity", "ellipticity", "intensity_factor"}\
                or name.endswith("_local"):
            # Ratios should be an exponential, as the analysis will log-transform the data
            value = math.exp(value)

        array[i] = value
    return array


def _view_cell_type_parameters(window: Window):
    file = dialog.prompt_load_file("Model file", [("Linear model", "linear_model_pickled.sav")])
    if file is None:
        return  # Cancelled
    with open(file, "rb") as handle:
        input_output = pickle.load(handle)
        scaler: StandardScaler = pickle.load(handle)
        regressor: LogisticRegression = pickle.load(handle)
    model_parameters = _ModelParameters(cell_type_mapping=input_output.cell_type_mapping,
                                        input_mapping=input_output.input_mapping, scaler=scaler, regressor=regressor)
    activate(_CellTypeParameterViewer(window, model_parameters))


class _CellParametersPopup(RichTextPopup):

    _experiment: Experiment
    _position: Position
    _parameters: _ModelParameters

    def __init__(self, experiment: Experiment, position: Position, parameters: _ModelParameters):
        self._experiment = experiment
        self._position = position
        self._parameters = parameters

    def get_title(self) -> str:
        return "Cell type information"

    def navigate(self, url: str) -> Optional[str]:
        if url == self.INDEX:
            return "# Cell position parameters\n" \
                   "In the training data, all measured parameters were scaled such that each has a mean of 0 and a " \
                   "standard deviation of 1. Here, we first apply the same scaling. Then we calculate the score of " \
                   "each cell type using the formula below. Then the chance of each cell type is `score / sum of " \
                   "scores`.\n\n"\
                + self._view_position_parameters()
        return None

    def _view_position_parameters(self) -> str:
        values = _get_data_array(self._experiment.position_data, self._position, self._parameters.input_mapping)
        if values is None:
            return "**We're missing some measurement values, so we cannot calculate the cell type scores.**"
        scaled_values = self._parameters.scaler.transform(values.reshape(1, -1)).flatten()

        output = ""
        for cell_type, coefficients, intercept in zip(self._parameters.cell_type_mapping,
                                                      self._parameters.regressor.coef_, self._parameters.regressor.intercept_):
            output += "## " + cell_type + "\n"
            output += "logit_score =  \n"
            logit_score = 0
            max_parameter_name = self._parameters.input_mapping[numpy.argmax(coefficients * scaled_values)]
            min_parameter_name = self._parameters.input_mapping[numpy.argmin(coefficients * scaled_values)]
            for parameter_name, coefficient, scaled_value in zip(self._parameters.input_mapping, coefficients,
                                                                 scaled_values):
                color = ""
                tag = "span"
                if abs(coefficient * scaled_value) > 0.5:
                    if coefficient * scaled_value > 0.5:
                        color = " style='color:#2ecc71'"  # Green
                    else:
                        color = " style='color:#8e44ad'"  # Purple
                    if parameter_name == max_parameter_name or parameter_name == min_parameter_name:
                        tag = "b"  # Most important contributor, make bold
                statement = "Lower value is better" if coefficient < 0 else "Higher value is better"
                output += f"<{tag}{color}>{coefficient:.4f} * {parameter_name} [{scaled_value:.4f}]</{tag}> + <small style='color:#dddddd'><i>{statement}</i></small>  \n"
                logit_score += coefficient * scaled_value
            logit_score += intercept
            output += f"{intercept:.4f} = {logit_score:.4f}\n\n"
            output += f"score = expit(logit_score) = {scipy.special.expit(logit_score):.4f}\n\n"
        return output


class _CellTypeParameterViewer(ExitableImageVisualizer):
    """For viewing information about how the model decides a cell type. Double-click a position to view the
    calculations."""

    _parameters: _ModelParameters

    def __init__(self, window: Window, parameters: _ModelParameters):
        super().__init__(window)
        self._parameters = parameters

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return
        selected_position = self._get_position_at(event.xdata, event.ydata)
        if selected_position is None:
            self.get_window().set_status("No cell found at position.")
            return

        dialog.popup_rich_text(_CellParametersPopup(self._experiment, selected_position, self._parameters))