from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.imaging import list_io
import lib_figures

_INPUT_FILE = r"../../Data/Training data.autlist"


def main():
    for experiment in list_io.load_experiment_list_file(_INPUT_FILE):
        if experiment.name.get_name() != "x20190926pos01":
            continue

        # Crop and project to our area of interest
        image_stack = experiment.images.get_image_stack(TimePoint(277), ImageChannel(index_zero=0))
        image_stack = image_stack[5:9]

        # Take brighest three pixels in the depth
        # image_stack.partition(2, axis=0)
        # image_stack = image_stack[2:]

        image = image_stack.mean(axis=0) + image_stack.max(axis=0) * 0.25

        figure = lib_figures.new_figure()
        ax: Axes = figure.gca()
        ax.imshow(image[290:460,70:335], cmap="gray")
        ax.set_axis_off()
        plt.show()


if __name__ == "__main__":
    main()
