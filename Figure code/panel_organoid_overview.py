import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.imaging import list_io

_DATA_FILE = "../Data/All time lapses.autlist"
_NUCLEUS_CHANNEL = ImageChannel(index_zero=0)
_TRANSMITTED_CHANNEL = ImageChannel(index_zero=1)
_SEGMENTATION_CHANNEL = ImageChannel(index_zero=2)
_Z = 14


def main():
    experiment_names = list()
    nuclear_images = list()
    transmitted_images = list()
    segmentation_images = list()

    for experiment in list_io.load_experiment_list_file(_DATA_FILE):
        print(f"Working on {experiment.name}...")
        last_time_point = TimePoint(experiment.positions.last_time_point_number())

        nuclei = experiment.images.get_image_slice_2d(last_time_point, _NUCLEUS_CHANNEL, _Z)
        transmitted_light = experiment.images.get_image_slice_2d(last_time_point, _TRANSMITTED_CHANNEL, _Z)
        segmentation = experiment.images.get_image_slice_2d(last_time_point, _SEGMENTATION_CHANNEL, _Z)

        experiment_names.append(experiment.name.get_name() + "\n" + str(experiment.images.resolution().pixel_size_x_um))
        nuclear_images.append(nuclei)
        transmitted_images.append(transmitted_light)
        segmentation_images.append(segmentation)

    figure: Figure = plt.figure(figsize=(10, 4))
    axes = figure.subplots(nrows=3, ncols=len(nuclear_images))
    for i in range(len(nuclear_images)):
        axes[0][i].set_title(experiment_names[i])
        _show_image(axes[0][i], nuclear_images[i], cmap="gray")
        _show_image(axes[1][i], transmitted_images[i], cmap="gray")
        _show_image(axes[2][i], segmentation_images[i], cmap="gist_ncar")
    plt.show()


def _show_image(ax: Axes, array: ndarray, cmap: str):
    ax.imshow(array, cmap=cmap, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(100, 350)
    ax.set_ylim(100, 350)


if __name__ == "__main__":
    main()
