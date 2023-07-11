from typing import NamedTuple, Optional


class ClassifiedCellType(NamedTuple):
    identifier: str


UNKNOWN_CELL_TYPE = ClassifiedCellType(identifier="UNKNOWN")


def convert_cell_type(position_type: Optional[str], *, allow_guessing: bool = False):
    """Converts an OrganoidTracker cell type to one suitable for the position detector. If allow_guessing is true,
    then inferred cell types are converted too."""
    if position_type is None:
        return UNKNOWN_CELL_TYPE
    return ClassifiedCellType(position_type.upper())
