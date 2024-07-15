import os
import pickle
from typing import Optional, NamedTuple, List, Set, Dict

import numpy
import scipy
from matplotlib.backend_bases import MouseEvent
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from organoid_tracker.core import UserError
from organoid_tracker.gui.dialog import DefaultOption
from . import lib_data
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.text_popup.text_popup import RichTextPopup
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class _ModelParameters(NamedTuple):
    cell_type_mapping: List[str]
    input_mapping: List[str]
    scaler: StandardScaler
    regressor: LogisticRegression
    disabled_parameters: Set[str]


def view_cell_type_parameters(window: Window):
    folder = dialog.prompt_directory("Model folder")
    if folder is None:
        return  # Cancelled
    file = os.path.join(folder, "linear_model_pickled.sav")
    if not os.path.exists(file):
        raise UserError("Invalid folder", f"This folder doesn't contain a linear cell type prediction"
                        f" model.\n\n(File {os.path.abspath(file)} not found.)")
    with open(file, "rb") as handle:
        input_output = pickle.load(handle)
        scaler: StandardScaler = pickle.load(handle)
        regressor: LogisticRegression = pickle.load(handle)
    model_parameters = _ModelParameters(cell_type_mapping=input_output["cell_type_mapping"],
                                        input_mapping=input_output["input_mapping"], scaler=scaler, regressor=regressor,
                                        disabled_parameters=set())
    activate(_CellTypeParameterViewer(window, model_parameters))


def _save_model(parameters: _ModelParameters):
    if not dialog.popup_message_cancellable("Model saving", "This will create a new model with the weight of"
                                            " the disabled parameters set to 0."):
        return
    regressor = parameters.regressor
    regressor = pickle.loads(pickle.dumps(regressor))  # Hack to make a copy, so that we can change the coefficients

    # Change the coefficients
    for disabled_parameter in parameters.disabled_parameters:
        parameter_index = parameters.input_mapping.index(disabled_parameter)
        regressor.coef_[:, parameter_index] = 0

    # Write to file
    folder = dialog.prompt_save_file("Model", [("Folder", "*")])
    if folder is None:
        return
    os.makedirs(folder, exist_ok=True)
    file = os.path.join(folder, "linear_model_pickled.sav")
    with open(file, "wb") as handle:
        pickle.dump({"cell_type_mapping": parameters.cell_type_mapping, "input_mapping": parameters.input_mapping},
                    handle)
        pickle.dump(parameters.scaler, handle)
        pickle.dump(regressor, handle)
        pickle.dump(parameters.disabled_parameters, handle)
    if dialog.prompt_options("Model saved", f"The model has been saved to {folder}.",
                             option_1="Open that directory", option_default=DefaultOption.OK) == 1:
        dialog.open_file(folder)


class _CellParametersPopup(RichTextPopup):
    _window: Window
    _experiment: Experiment
    _position: Position
    _parameters: _ModelParameters

    def __init__(self, window: Window, experiment: Experiment, position: Position, parameters: _ModelParameters):
        self._window = window
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
        if url == "save":
            _save_model(self._parameters)
            return None
        if url.startswith("disable_parameter/"):
            feature_name = url[len("disable_parameter/"):]
            self._parameters.disabled_parameters.add(feature_name)
            self._window.redraw_data()  # Have the new cell types show up
            return self.navigate(self.INDEX)
        if url.startswith("reenable_parameter/"):
            feature_name = url[len("reenable_parameter/"):]
            self._parameters.disabled_parameters.remove(feature_name)
            self._window.redraw_data()  # Have the new cell types show up
            return self.navigate(self.INDEX)
        return None

    def _view_position_parameters(self) -> str:
        values = lib_data.get_data_array(self._experiment.position_data, self._position, self._parameters.input_mapping)
        if values is None:
            return "**We're missing some measurement values, so we cannot calculate the cell type scores.**"
        scaled_values = self._parameters.scaler.transform(values.reshape(1, -1)).flatten()
        numpy.clip(scaled_values, -3, 3, out=scaled_values)

        cell_type_scores = list()

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
                disable_parameter_link = f" <a href=\"disable_parameter/{parameter_name}\">Disable</a>"

                if abs(coefficient * scaled_value) > 0.5:
                    if coefficient * scaled_value > 0.5:
                        color = " style='color:#2ecc71'"  # Green
                    else:
                        color = " style='color:#8e44ad'"  # Purple
                    if parameter_name == max_parameter_name or parameter_name == min_parameter_name:
                        tag = "b"  # Most important contributor, make bold
                if coefficient == 0:
                    # Parameter that is not used, maybe as a result of the user saving the model
                    color = " style='color:#dddddd'"
                    disable_parameter_link = ""

                if parameter_name in self._parameters.disabled_parameters:
                    # Disable this feature
                    color = " style='color:#dddddd'"
                    tag = "s"
                    disable_parameter_link = f" <a href=\"reenable_parameter/{parameter_name}\">Re-enable</a>"

                statement = "Lower value is better" if coefficient < 0 else (
                    "Higher value is better" if coefficient > 0 else "No influence on this cell type")
                output += f"<{tag}{color}>{coefficient:.4f} * {parameter_name} [{scaled_value:.4f}] +</{tag}> <small style='color:#dddddd'><i>{statement}{disable_parameter_link}</i></small>  \n"

                if parameter_name not in self._parameters.disabled_parameters:
                    logit_score += coefficient * scaled_value
            logit_score += intercept
            output += f"{intercept:.4f} = {logit_score:.4f}\n\n"
            output += f"score = expit(logit_score) = {scipy.special.expit(logit_score):.4f}\n\n"
            cell_type_scores.append(scipy.special.expit(logit_score))

        best_cell_type = self._parameters.cell_type_mapping[numpy.argmax(cell_type_scores)]

        preface = f"## Most likely cell type: {best_cell_type}\n\n"
        if self._parameters.disabled_parameters:
            preface += (f"Disabled parameters: `{', '.join(self._parameters.disabled_parameters)}`."
                        f" [Save model](save).\n\n")
        else:
            preface += "If you get unexpected cell types, you can disable parameters using the links below.\n\n"

        return preface + output


class _CellTypeParameterViewer(ExitableImageVisualizer):
    """For viewing information about how the model decides a cell type. Double-click a position to view the
    calculations."""

    _parameters: _ModelParameters
    _cell_types_by_position: Dict[Position, str]

    def __init__(self, window: Window, parameters: _ModelParameters):
        super().__init__(window)
        self._parameters = parameters
        self._cell_types_by_position = dict()

    def _on_mouse_single_click(self, event: MouseEvent):
        selected_position = self._get_position_at(event.xdata, event.ydata)
        if selected_position is None:
            self.get_window().set_status("No cell found at position.")
            return

        dialog.popup_rich_text(_CellParametersPopup(self._window, self._experiment, selected_position, self._parameters))

    def _calculate_time_point_metadata(self):
        self._cell_types_by_position.clear()
        for position in self._experiment.positions.of_time_point(self._time_point):
            self._cell_types_by_position[position] = self._get_updated_type(position)

    def _get_position_edge(self, position: Position) -> tuple[tuple[float, float, float], float]:
        """We use our own cell type lookup."""
        position_type = self.get_window().registry.get_marker_by_save_name(self._cell_types_by_position.get(position))
        edge_color = (0.0, 0.0, 0.0) if position_type is None else position_type.mpl_color
        edge_width = 1.0 if position_type is None else 3.0
        return edge_color, edge_width

    def _get_updated_type(self, position: Position) -> Optional[str]:
        values = lib_data.get_data_array(self._experiment.position_data, position, self._parameters.input_mapping)
        if values is None:
            return None
        scaled_values = self._parameters.scaler.transform(values.reshape(1, -1)).flatten()
        numpy.clip(scaled_values, -3, 3, out=scaled_values)

        cell_type_logit_scores = list()
        for cell_type, coefficients, intercept in zip(self._parameters.cell_type_mapping,
                                                      self._parameters.regressor.coef_, self._parameters.regressor.intercept_):
            logit_score = 0
            for parameter_name, coefficient, scaled_value in zip(self._parameters.input_mapping, coefficients,
                                                                 scaled_values):
                if parameter_name not in self._parameters.disabled_parameters:
                    logit_score += coefficient * scaled_value
            logit_score += intercept
            cell_type_logit_scores.append(logit_score)

        best_cell_type = self._parameters.cell_type_mapping[numpy.argmax(cell_type_logit_scores)]
        return best_cell_type
