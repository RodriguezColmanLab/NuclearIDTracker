import numpy
import skimage
from matplotlib import pyplot as plt
from typing import NamedTuple

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.imaging import list_io, cropper
from organoid_tracker.linking import nearby_position_finder


class _CellToRender(NamedTuple):
    experiment_name: str
    position: Position


_DATASET_FILE = "../../Data/Training data.autlist"
_CELLS_TO_RENDER = [
    # Goblet cell:
    _CellToRender("Interleukin_10IL_9CTRL_xy05", Position(223.00, 268.00, 24.00, time_point_number=100)),
    # Stem cell:
    _CellToRender("x20190817pos01", Position(336.86, 334.11, 3.00, time_point_number=313)),
    # Paneth cell:
    _CellToRender("x20190817pos01", Position(295.19, 337.13, 5.00, time_point_number=313)),
    # Enterocyte:
    _CellToRender("x20190817pos01", Position(319.10, 110.53, 14.00, time_point_number=313))
]
_NUCLEUS_CHANNEL = ImageChannel(index_one=1)
_SEGMENTATION_CHANNEL = ImageChannel(index_one=3)
_CROP_WIDTH_PX = 70


_RENDER_PYVISTA = False
try:
    import pyvista
    _RENDER_PYVISTA = True
except ImportError:
    print("PyVista not installed, skipping 3D rendering.")


def main():
    for experiment in list_io.load_experiment_list_file(_DATASET_FILE):
        resolution = experiment.images.resolution()
        z_scale = int(resolution.pixel_size_z_um / resolution.pixel_size_x_um)

        for cell in _CELLS_TO_RENDER:
            if cell.experiment_name != experiment.name.get_name():
                continue
            if not experiment.positions.contains_position(cell.position):
                raise ValueError(f"Position {cell.position} not found in experiment {cell.experiment_name}")

            nuclei_image = experiment.images.get_image(cell.position.time_point(), _NUCLEUS_CHANNEL)
            if nuclei_image is None:
                raise ValueError(f"Nucleus image not found for {cell.experiment_name} at {cell.position}")
            segmentation_image = experiment.images.get_image(cell.position.time_point(), _SEGMENTATION_CHANNEL)
            if segmentation_image is None:
                raise ValueError(f"Segmentation image not found for {cell.experiment_name} at {cell.position}")


            # Make 2D image
            nuclei_image_slice = nuclei_image.get_image_slice_2d(round(cell.position.z))
            image_position = cell.position - nuclei_image.offset
            nuclei_image_slice = nuclei_image_slice[
                          int(image_position.y) - _CROP_WIDTH_PX // 2:int(image_position.y) + _CROP_WIDTH_PX // 2,
                          int(image_position.x) - _CROP_WIDTH_PX // 2:int(image_position.x) + _CROP_WIDTH_PX // 2,
            ]

            plt.imshow(nuclei_image_slice, cmap="gray")
            plt.show()

            # Make 3D render
            if _RENDER_PYVISTA:
                _render_pyvista(cell, segmentation_image, z_scale)

            # Make mean projection
            _render_segmentation(cell, experiment, segmentation_image)



def _find_nearest_neighbors(experiment: Experiment, position: Position, num_neighbors: int):
    return nearby_position_finder.find_closest_n_positions(experiment.positions.of_time_point(position.time_point()),
                                                    around=position, max_amount=num_neighbors, resolution=experiment.images.resolution())


def _erode_each_label(crop_array: numpy.ndarray):
    circle_mask = skimage.morphology.disk(1)
    for label in numpy.unique(crop_array):
        if label == 0:
            continue
        mask = crop_array == label
        mask_eroded = skimage.morphology.erosion(mask, footprint=circle_mask)

        # Replace the original mask with the eroded mask
        crop_array[mask] = 0
        crop_array[mask_eroded] = label


def _render_segmentation(cell: _CellToRender, experiment: Experiment, segmentation_image: Image, *,
                         xy_crop_size: int = _CROP_WIDTH_PX):
    x_start = int(round(cell.position.x)) - xy_crop_size // 2
    y_start = int(round(cell.position.y)) - xy_crop_size // 2
    z = int(round(cell.position.z))

    crop_array = numpy.zeros((xy_crop_size, xy_crop_size), dtype=numpy.uint16)
    cropper.crop_3d_to_2d(segmentation_image, x_start, y_start, z, crop_array)

    _erode_each_label(crop_array)

    # Change to RGB image
    rgb_image = numpy.zeros((xy_crop_size, xy_crop_size, 3), dtype=numpy.uint8)
    rgb_image[...] = 255  # Set background to white

    # Set other cells to gray
    rgb_image[crop_array > 0] = [180, 206, 211]

    # Set nearest neighbors to green
    neighbors = _find_nearest_neighbors(experiment, cell.position, num_neighbors=6)
    for neighbor in neighbors:
        neighbor_label = segmentation_image.value_at(neighbor)
        rgb_image[crop_array == neighbor_label] = [133, 255, 142]

    # Set cell of interest to blue
    cell_of_interest = segmentation_image.value_at(cell.position)
    rgb_image[crop_array == cell_of_interest] = [120, 120, 231]

    # Render cell centers
    x_points = []
    y_points = []
    for position in experiment.positions.of_time_point(cell.position.time_point()):
        if (abs(position.x - cell.position.x) > xy_crop_size // 2
                or abs(position.y - cell.position.y) > xy_crop_size // 2
                or abs(position.z - cell.position.z) > 2):
            continue
        x_points.append(int(round(position.x)) - x_start)
        y_points.append(int(round(position.y)) - y_start)

    figure = plt.figure(figsize=(5, 5))
    ax = figure.gca()
    ax.imshow(rgb_image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.scatter(x_points, y_points, color="#2d3436", s=20, lw=0, marker="s")
    plt.show()


def _render_pyvista(cell: _CellToRender, segmentation_image: Image, z_scale: int):
    image_position = cell.position - segmentation_image.offset

    segmentation_array = segmentation_image.array
    cell_label = segmentation_array[int(round(image_position.z)), int(image_position.y), int(image_position.x)]
    if cell_label == 0:
        raise ValueError(f"Cell label not found for {cell.experiment_name} at {cell.position}")

    # Pad the array to avoid edge effects
    pad_width = z_scale
    segmentation_array = numpy.pad(segmentation_array, pad_width=pad_width, mode='constant', constant_values=0)

    # Flip Z axis
    segmentation_array = numpy.flip(segmentation_array, axis=0)
    # Flip Y axes
    segmentation_array = numpy.flip(segmentation_array, axis=1)
    # Flip X axes
    # segmentation_array = numpy.flip(segmentation_array, axis=2)

    plotter = pyvista.Plotter()

    # Loop over all cells using regionprops
    # and create a mesh for each cell

    color_index = 0
    for region in skimage.measure.regionprops(segmentation_array):
        z_start, y_start, x_start, z_end, y_end, x_end = region.bbox
        z_start -= 1
        y_start -= 1
        x_start -= 1
        z_end += 1
        y_end += 1
        x_end += 1

        grid = pyvista.ImageData(
            dimensions=((x_end - x_start), (y_end - y_start), (z_end - z_start) * z_scale),
            spacing=(1, 1, 1),
            origin=(x_start, y_start, z_start * z_scale),
        )
        x, y, z = grid.points.T
        x = x.astype(numpy.int32)
        y = y.astype(numpy.int32)
        z = (z / z_scale).astype(numpy.int32)
        values = numpy.where(segmentation_array[z, y, x] == region.label, 2, 0)

        mesh = grid.contour([1], values, method='marching_cubes')
        if mesh.n_points == 0:
            continue
        mesh.decimate(target_reduction=0.5, inplace=True)
        mesh.smooth_taubin(n_iter=200, pass_band=0.05, inplace=True)
        mesh.subdivide(1, subfilter="loop", inplace=True)
        mesh.smooth(n_iter=200, inplace=True)

        plotter.add_mesh(mesh, color="#ffffff" if region.label != cell_label else "#ff0000")
        color_index += 1
    plotter.background_color = "black"
    plotter.show()


if __name__ == "__main__":
    main()