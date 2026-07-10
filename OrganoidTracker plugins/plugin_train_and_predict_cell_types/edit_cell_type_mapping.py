"""Used to select which cell types to use for training. The training will be done on the selected cell types, and
 others will be ignored."""
from . import lib_data

from organoid_tracker.core import UserError
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.text_popup.text_popup import RichTextPopup



def show_popup(window: Window):
    """Shows the popup to edit cell types."""
    dialog.popup_rich_text(CellTypeMappingPopup(window))


class CellTypeMappingPopup(RichTextPopup):

    _window: Window

    def __init__(self, window: Window):
        self._window = window

    def _get_cell_types(self) -> list[str]:
        """Returns the cell types that are present in the current experiments."""
        return [marker.save_name for marker in self._window.registry.get_registered_markers(Position)]

    def _get_cell_type_mapping(self) -> dict[str, str]:
        """Returns the mapping from cell types to their merged names."""

        # Find a suitable experiment to grab the mappings of
        try:
            experiment = self._window.get_experiment()
        except UserError:
            experiment = None
            for some_experiment in self._window.get_active_experiments():
                experiment = some_experiment
                break
            if experiment is None:
                return dict()  # No data loaded

        return lib_data.get_cell_type_mapping(experiment)

    def _set_cell_type_mapping(self, mapping: dict[str, str]):
        for tab in self._window.get_gui_experiment().get_active_tabs():
            lib_data.set_cell_type_mapping(tab.experiment, mapping)
            tab.undo_redo.mark_unsaved_changes()

    def get_title(self) -> str:
        """Returns the title for the whole website."""
        return "Cell type mapping"

    def navigate(self, url: str) -> str | None:
        """Returns the Markdown for the given URL. The website can also perform some other action, like zooming in on a
        position or starting a program. If this method returns None, then the navigation is cancelled."""
        if url == self.INDEX:
            return self._home_page()
        if url.startswith("change_mapping/"):
            self._change_cell_type(url[len("change_mapping/"):])
            return self._home_page()
        raise UserError("Unknown URL", f"Cannot handle '{url}'")

    def _change_cell_type(self, cell_type: str):
        options = self._get_cell_types()
        options.sort()

        # Put own cell type on top
        options.remove(cell_type)
        options.insert(0, cell_type)
        options.append("NONE")

        option_names = ["convert to " + option for option in options]
        option_names[0] = "« just use as itself »"
        option_names[-1] = "« don't use »"

        result_index = option_choose_dialog.prompt_list(f"Mapping of '{cell_type}'",
                                         f"How should we use '{cell_type}' during training?",
                                         "Mapped cell type:", option_names)
        if result_index is None:
            return  # Cancelled

        new_mapping = self._get_cell_type_mapping()
        result = options[result_index]
        print("Mapping", cell_type, "to", result)
        if result == "NONE":
            if cell_type in new_mapping:
                del new_mapping[cell_type]
        else:
            new_mapping[cell_type] = result
        self._set_cell_type_mapping(new_mapping)


    def _home_page(self) -> str:
        """Returns the Markdown for the home page."""
        message = ("## Cell type mapping\n\n"
                 "You don't have to use all cell types for your machine learning model. You can merge several cell"
                 " types into one, or ignore some cell types. This is useful if you have a lot of cell types that"
                 " look alike, but only a few of them are relevant for your analysis.\n\n")

        cell_types = self._get_cell_types()
        cell_type_mapping = self._get_cell_type_mapping()
        if len(cell_types) == 0:
            message += "Right now, there are no cell types defined. Install some plugins for cell types to get started."
        else:
            for cell_type in cell_types:
                mapped_type = cell_type_mapping.get(cell_type)
                if mapped_type is None:
                    message += f"* `{cell_type}` - not used"
                elif mapped_type == cell_type:
                    message += f"* `{cell_type}` - **used**"
                else:
                    message += f"* `{cell_type}` - **converted** to `{mapped_type}`"
                message += f" *[change](change_mapping/{cell_type})*\n"
        return message
