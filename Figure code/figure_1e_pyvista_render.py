import numpy
import pyvista
import skimage
import tifffile

_IMAGE_TO_RENDER = r"P:\Rodriguez_Colman\vidi_rodriguez_colman\rkok\data_analysis\2023\2023-05 RK0029 Rutger Measuring CellPose performance\ActiveUnet segmentation\20190926pos01_t236_c1.tif"
_Z_SCALE = 5
_COLORS = ["#D21820", "#1869FF", "#008A00", "#F36DFF", "#710079", "#AAFB00", "#00BEC2", "#FFA235", "#5D3D04", "#08008A",
           "#005D5D", "#9A7D82", "#A2AEFF", "#96B675", "#9E28FF", "#FFAEBE", "#CE0092", "#00FFB6", "#9E7500", "#3D3541",
           "#F3EB92", "#65618A", "#8A3D4D"]


def main():
    # Load the file
    array = tifffile.imread(_IMAGE_TO_RENDER)

    # Pad the array to avoid edge effects
    array = numpy.pad(array, pad_width=_Z_SCALE, mode='constant', constant_values=0)

    plotter = pyvista.Plotter()

    # Loop over all cells using regionprops
    # and create a mesh for each cell
    color_index = 0
    for cell in skimage.measure.regionprops(array):
        z_start, y_start, x_start, z_end, y_end, x_end = cell.bbox
        z_start -= 1
        y_start -= 1
        x_start -= 1
        z_end += 1
        y_end += 1
        x_end += 1

        grid = pyvista.ImageData(
            dimensions=((x_end - x_start), (y_end - y_start), (z_end - z_start) * _Z_SCALE),
            spacing=(1, 1, 1),
            origin=(x_start, y_start, z_start * _Z_SCALE),
        )
        x, y, z = grid.points.T
        x = x.astype(numpy.int32)
        y = y.astype(numpy.int32)
        z = (z / _Z_SCALE).astype(numpy.int32)
        values = numpy.where(array[z, y, x] == cell.label, 2, 0)

        mesh = grid.contour([1], values, method='marching_cubes')
        if mesh.n_points == 0:
            continue
        mesh.decimate(target_reduction=0.5, inplace=True)
        mesh.smooth_taubin(n_iter=200, pass_band=0.05, inplace=True)
        mesh.smooth(n_iter=200, inplace=True)

        plotter.add_mesh(mesh, color=_COLORS[color_index % len(_COLORS)])
        color_index += 1
    plotter.background_color = "black"
    plotter.show()


if __name__ == "__main__":
    main()