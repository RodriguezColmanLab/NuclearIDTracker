from typing import NamedTuple, Optional


class ClassifiedCellType(NamedTuple):
    identifier: str


UNKNOWN_CELL_TYPE = ClassifiedCellType(identifier="UNKNOWN")


def convert_cell_type(position_type: Optional[str], *, allow_guessing: bool = False):
    """Converts an OrganoidTracker cell type to one suitable for the position detector. If allow_guessing is true,
    then inferred cell types are converted too."""
    if position_type is None:
        return UNKNOWN_CELL_TYPE
    if position_type in {"WGA_PLUS", "ENTEROENDOCRINE", "GOBLET", "TUFT", "SECRETORY"}:
        # Seeing the difference between these is hard for the network
        return ClassifiedCellType("secretory")
    if position_type == "PANETH":
        return ClassifiedCellType("paneth")
    if position_type in {"STEM", "STEM_PUTATIVE"}:
        return ClassifiedCellType("stem")
    if position_type == "ENTEROCYTE":
        return ClassifiedCellType("enterocyte")

    if allow_guessing:
        if position_type == "UNLABELED":
            return ClassifiedCellType("stem")
        if position_type == "ABSORPTIVE_PRECURSOR":
            return ClassifiedCellType("enterocyte")
        if position_type == "SECRETIVE_PRECURSOR":
            return ClassifiedCellType("secretory")
        if position_type == "PANETH_PRECURSOR":
            return ClassifiedCellType("paneth")

    return UNKNOWN_CELL_TYPE
