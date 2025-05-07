"""We plot all stem and Paneth cells on a stem-to-Paneth axis, and see how their amount changes over time."""
from enum import Enum, auto
from typing import List

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
from organoid_tracker.core import Name
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE = "../../Data/Live and immunostaining multiple time points/Immunostaining young versus old.autlist"


class _Age(Enum):
    YOUNG = auto()
    OLD = auto()

    def get_age_text(self) -> str:
        if self == _Age.YOUNG:
            return "Day 2"
        elif self == _Age.OLD:
            return "Day 5"
        else:
            raise ValueError(f"Unknown age {self}")


class _StainingCounts:
    lgr5_positive_count: int = 0
    krt20_positive_count: int = 0
    ki67_positive_count: int = 0
    wga_positive_count: int = 0

    def add_cell(self, type: str):
        if type == "STEM_PUTATIVE":
            self.ki67_positive_count += 1
        elif type == "STEM":
            self.lgr5_positive_count += 1
        elif type == "ENTEROCYTE":
            self.krt20_positive_count += 1
        elif type == "ABSORPTIVE_PRECURSOR":
            self.krt20_positive_count += 1
            self.ki67_positive_count += 1
        elif type == "PANETH":
            self.wga_positive_count += 1
        else:
            raise ValueError(f"Unknown cell type {type}")

    def __str__(self) -> str:
        return "{" + f"lgr5_positive_count: {self.lgr5_positive_count}, " \
               f"krt20_positive_count: {self.krt20_positive_count}, " \
               f"ki67_positive_count: {self.ki67_positive_count}" + "}"

    def has_stainings(self) -> bool:
        return self.lgr5_positive_count + self.krt20_positive_count + self.ki67_positive_count + self.wga_positive_count > 0

    @property
    def ta_count(self) -> int:
        return self.ki67_positive_count - self.lgr5_positive_count


def _get_age(experiment_name: Name) -> _Age:
    if "A1 »" in experiment_name.get_name():
        return _Age.YOUNG  # This well contained young organoids
    if "B2 »" in experiment_name.get_name():
        return _Age.OLD  # This well contained old organoids
    raise ValueError(f"Cannot determine age from experiment name {experiment_name}")


def _plot_stem_cell_counts(ax: Axes, stainings: List[_StainingCounts]):
    bottom = numpy.zeros(len(stainings), dtype=numpy.uint32)
    x_positions = numpy.arange(1, len(stainings) + 1)

    ta_counts = [staining.ta_count for staining in stainings]
    ax.bar(
        x_positions,
        ta_counts,
        label="TA (KI67+, LGR5-)",
        color=lib_figures.CELL_TYPE_PALETTE["UNLABELED"])
    bottom += numpy.array(ta_counts, dtype=numpy.uint32)

    lgr5_positive_counts = [staining.lgr5_positive_count for staining in stainings]
    ax.bar(
        x_positions,
        lgr5_positive_counts,
        label="Stem (LGR5+)",
        color=lib_figures.CELL_TYPE_PALETTE["STEM"],
        bottom=bottom
    )
    bottom += numpy.array(lgr5_positive_counts, dtype=numpy.uint32)

    wga_positive_counts = [staining.wga_positive_count for staining in stainings]
    ax.bar(
        x_positions,
        wga_positive_counts,
        label="Paneth (WGA+)",
        color=lib_figures.CELL_TYPE_PALETTE["PANETH"],
        bottom=bottom
    )
    bottom += numpy.array(wga_positive_counts, dtype=numpy.uint32)

    enterocyte_counts = [staining.krt20_positive_count for staining in stainings]
    ax.bar(
        x_positions,
        enterocyte_counts,
        label="Enterocyte (KRT20+)",
        color=lib_figures.CELL_TYPE_PALETTE["ENTEROCYTE"],
        bottom=bottom
    )


def main():

    stainings_young = list()
    stainings_old = list()

    _count_stainings(stainings_old, stainings_young)

    # Plotting
    figure = lib_figures.new_figure()
    ax_young, ax_old = figure.subplots(nrows=1, ncols=2, sharey=True)
    ax_young.set_title(_Age.YOUNG.get_age_text())
    ax_old.set_title(_Age.OLD.get_age_text())
    ax_young.set_ylabel("Cell count")
    ax_young.set_xlabel("Organoid")
    ax_old.set_xlabel("Organoid")

    _plot_stem_cell_counts(ax_young, stainings_young)
    _plot_stem_cell_counts(ax_old, stainings_old)

    # Hide right and top spines
    ax_young.spines["right"].set_visible(False)
    ax_old.spines["right"].set_visible(False)
    ax_young.spines["top"].set_visible(False)
    ax_old.spines["top"].set_visible(False)
    ax_young.set_xlim(0.25, len(stainings_young) + 0.75)
    ax_old.set_xlim(0.25, len(stainings_old) + 0.75)

    # Place legend next to ax_old
    ax_old.legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=ax_old.transAxes
    )

    figure.tight_layout()
    plt.show()


def _count_stainings(stainings_old, stainings_young):
    for experiment in list_io.load_experiment_list_file(_DATA_FILE, load_images=False):
        position_data = experiment.position_data

        stainings = _StainingCounts()
        for position in experiment.positions:
            cell_type = position_markers.get_position_type(position_data, position)
            if cell_type is None:
                continue
            stainings.add_cell(cell_type)
        if not stainings.has_stainings():
            continue

        age = _get_age(experiment.name)
        if age == _Age.YOUNG:
            stainings_young.append(stainings)
        elif age == _Age.OLD:
            stainings_old.append(stainings)
        else:
            raise ValueError(f"Unknown age {age}")


if __name__ == "__main__":
    main()

