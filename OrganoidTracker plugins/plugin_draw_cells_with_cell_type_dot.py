import numpy
import skimage
from matplotlib import pyplot as plt
from numpy import ndarray

from organoid_tracker.core import UserError, Color
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.util import bits


def get_menu_items(window: Window):
    return {
        "File//Export-Export image//Projection-Max-intensity with cell type dots...": lambda: _show_colored_image(window),
        #"File//Export-Export movie//Projection-Max-intensity with cell type dots...": lambda: _show_colored_movie(window)
    }


def _draw_marker(image_rgb: ndarray, image_x: float, image_y: float, color: Color):
    rr, cc = skimage.draw.disk((int(image_y), int(image_x)), 7, shape=image_rgb.shape[0:2])
    image_rgb[rr, cc, :] = (color.red, color.green, color.blue)

def _show_colored_image(window: Window):
    experiment = window.get_experiment()
    time_point = window.display_settings.time_point

    image = experiment.images.get_image_stack(time_point, window.display_settings.image_channel)
    if image is None:
        raise UserError("No image found", "No image found at the given time point")
    image = image.max(axis=0)
    image = bits.ensure_8bit(image)
    image_rgb = numpy.zeros(image.shape + (3,), dtype=numpy.uint8)
    for i in range(3):
        image_rgb[:, :, i] = image

    image_offset = experiment.images.offsets.of_time_point(time_point)
    position_data = experiment.position_data
    for position in experiment.positions.of_time_point(time_point):
        marker = window.registry.get_marker_by_save_name(position_markers.get_position_type(position_data, position))
        if marker is None:
            continue
        _draw_marker(image_rgb, position.x - image_offset.x, position.y - image_offset.y, marker.color)

    output_file = dialog.prompt_save_file("Image", [("PNG file", "*.png")])
    if output_file is None:
        return
    plt.imsave(output_file, image_rgb)
