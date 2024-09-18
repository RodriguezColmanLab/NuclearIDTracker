# We follow the evaluation method of CellPose, so on the X-axis IoU from 0.5 to 1, and on the y axis average precision.

import os
from typing import Tuple, List, Optional, Iterable, NamedTuple

import numpy
import scipy
import skimage.measure
import tifffile
from matplotlib import pyplot as plt
from numpy import ndarray

import lib_figures

_GROUND_TRUTH_FOLDER = r"P:\Rodriguez_Colman\vidi_rodriguez_colman\rkok\data_analysis\2023\2023-05 RK0029 Rutger Measuring CellPose performance\Manual segmentation"
_AUTOMATIC_FOLDER = r"P:\Rodriguez_Colman\vidi_rodriguez_colman\rkok\data_analysis\2023\2023-05 RK0029 Rutger Measuring CellPose performance\ActiveUnet segmentation"
_NUCLEUS_FOLDER = r"P:\Rodriguez_Colman\vidi_rodriguez_colman\rkok\data_analysis\2023\2023-05 RK0029 Rutger Measuring CellPose performance\Nucleus images"
_PICKED_IOU = {40, 80}


class _IntersectionResult(NamedTuple):
    ground_truth_label: int
    automatic_label: int
    bbox: Tuple[int, int, int, int, int, int]  # Bbox is min_z, min_y, min_x, max_z, max_y, max_x
    iou: float


class _GroundTruthImage:
    """For measuring the overlap in a single image."""
    _regionprops: List[Optional["skimage.measure._regionprops.RegionProperties"]]
    _bboxes_zyx_zyx: ndarray  # 2D array, each row represents a bounding box

    def __init__(self, ground_truth: ndarray):
        max_region_id = ground_truth.max() + 1
        self._bboxes_zyx_zyx = numpy.zeros((max_region_id, 6))
        self._regionprops = [None] * max_region_id

        for region in skimage.measure.regionprops(ground_truth):
            self._bboxes_zyx_zyx[region.label] = region.bbox
            self._regionprops[region.label] = region

    def _get_overlapping_bounding_box(self, bounding_box: Tuple[int, int, int, int, int, int]) -> List[int]:
        overlapping = (bounding_box[5] >= self._bboxes_zyx_zyx[:, 2]) & (
                self._bboxes_zyx_zyx[:, 5] >= bounding_box[2]) & \
                      (bounding_box[4] >= self._bboxes_zyx_zyx[:, 1]) & (
                              self._bboxes_zyx_zyx[:, 4] >= bounding_box[1]) & \
                      (bounding_box[3] >= self._bboxes_zyx_zyx[:, 0]) & (self._bboxes_zyx_zyx[:, 3] >= bounding_box[0])
        return list(numpy.nonzero(overlapping)[0])

    def get_intersection_of_union(self, automatic_nucleus: "skimage.measure._regionprops.RegionProperties"
                                  ) -> Optional[_IntersectionResult]:
        """Gets the intersection-over-union of the automatic nucleus with the corresponding ground truth nucleus. If
        the automatic nucleus is a false positive, then 0 will be returned. If it overlaps with multiple nuclei in the
        ground truth, then the highest IoU is returned.

        Returns the intersection of union, as well as the ground truth nucleus that it intersected with."""
        max_iou = 0
        with_label = None
        with_bbox = None
        for overlapper_id in self._get_overlapping_bounding_box(automatic_nucleus.bbox):
            overlapper = self._regionprops[overlapper_id]

            # First, construct a bounding box
            bbox_union = (  # Bbox is min_z, min_y, min_x, max_z, max_y, max_x
                min(overlapper.bbox[0], automatic_nucleus.bbox[0]),
                min(overlapper.bbox[1], automatic_nucleus.bbox[1]),
                min(overlapper.bbox[2], automatic_nucleus.bbox[2]),
                max(overlapper.bbox[3], automatic_nucleus.bbox[3]),
                max(overlapper.bbox[4], automatic_nucleus.bbox[4]),
                max(overlapper.bbox[5], automatic_nucleus.bbox[5])
            )

            # Then, put masks of both nuclei into this bounding box
            mask_overlapper = numpy.zeros((bbox_union[3] - bbox_union[0],
                                           bbox_union[4] - bbox_union[1],
                                           bbox_union[5] - bbox_union[2]), dtype=bool)
            _place_at(mask_overlapper, (overlapper.bbox[0] - bbox_union[0],
                                        overlapper.bbox[1] - bbox_union[1],
                                        overlapper.bbox[2] - bbox_union[2]),
                      overlapper.image)
            mask_automatic = numpy.zeros_like(mask_overlapper)
            _place_at(mask_automatic, (automatic_nucleus.bbox[0] - bbox_union[0],
                                       automatic_nucleus.bbox[1] - bbox_union[1],
                                       automatic_nucleus.bbox[2] - bbox_union[2]),
                      automatic_nucleus.image)

            # Then, calculate intersection and union
            union = numpy.sum(numpy.logical_or(mask_overlapper, mask_automatic))
            intersection = numpy.sum(numpy.logical_and(mask_overlapper, mask_automatic))
            if intersection / union > max_iou:
                max_iou = float(intersection / union)
                with_label = overlapper.label
                with_bbox = bbox_union
        if with_label is None or with_bbox is None:
            return None
        return _IntersectionResult(ground_truth_label=with_label, automatic_label=automatic_nucleus.label,
                                   bbox=with_bbox, iou=max_iou)

    def get_nucleus_ids_and_z(self) -> Iterable[Tuple[int, float]]:
        """Gets all nucleus ids that are in the ground truth set."""
        for regionprops in self._regionprops:
            if regionprops is not None:
                yield regionprops.label, regionprops.centroid[0]


def _place_at(into: ndarray, start_zyx: Tuple[int, int, int], image: ndarray):
    into[start_zyx[0]:start_zyx[0] + image.shape[0],
    start_zyx[1]:start_zyx[1] + image.shape[1],
    start_zyx[2]:start_zyx[2] + image.shape[2]] = image


def _find_examples(ground_truth: ndarray, automatic: ndarray) -> Iterable[_IntersectionResult]:
    """Calculates the intersection over union for all nuclei, so that you can later on calculate the true/false
     positives and false negatives."""
    ground_truth_masks = _GroundTruthImage(ground_truth)

    for automatic_nucleus in skimage.measure.regionprops(automatic):
        result = ground_truth_masks.get_intersection_of_union(automatic_nucleus)
        if result is None:
            continue
        if round(result.iou * 100) in _PICKED_IOU:
            yield result


def main():
    for file in os.listdir(_GROUND_TRUTH_FOLDER):
        if not file.endswith(".tif"):
            continue

        ground_truth = tifffile.imread(os.path.join(_GROUND_TRUTH_FOLDER, file))
        ground_truth = _dilate_masks(ground_truth)
        automatic = tifffile.imread(os.path.join(_AUTOMATIC_FOLDER, file))
        nuclei = tifffile.imread(os.path.join(_NUCLEUS_FOLDER, file))
        for result in _find_examples(ground_truth, automatic):
            figure = lib_figures.new_figure()
            ax_segmentation, ax_nuclei = figure.subplots(nrows=1, ncols=2)

            margin = 5
            min_x, max_x = max(result.bbox[2] - margin, 0), result.bbox[5] + margin
            min_y, max_y = max(result.bbox[1] - margin, 0), result.bbox[4] + margin

            z = (result.bbox[0] + result.bbox[3]) // 2
            nucleus_crop = nuclei[z, min_y:max_y, min_x:max_x]

            ground_truth_crop = ground_truth[z, min_y:max_y, min_x:max_x] == result.ground_truth_label
            automatic_crop = automatic[z, min_y:max_y, min_x:max_x] == result.automatic_label
            segmentation_image = numpy.zeros(automatic_crop.shape + (3,), dtype=numpy.float32)
            segmentation_image[:, :][ground_truth_crop & ~automatic_crop] = rgb(214, 48, 49)  # Only ground truth
            segmentation_image[:, :][ground_truth_crop & automatic_crop] = rgb(9, 132, 227)  # Overlap
            segmentation_image[:, :][~ground_truth_crop & automatic_crop] = rgb(108, 92, 231)  # Only prediction

            ax_nuclei.imshow(nucleus_crop, cmap="gray")
            ax_segmentation.imshow(segmentation_image)
            ax_nuclei.axis("off")
            ax_segmentation.axis("off")
            figure.suptitle(f"IoU: {result.iou:.2f}")

            plt.show()


def _dilate_masks(ground_truth: ndarray):
    expanded = scipy.ndimage.maximum_filter(ground_truth, footprint=scipy.ndimage.generate_binary_structure(3, 1))
    expanded[ground_truth != 0] = ground_truth[ground_truth != 0]  # Leave original labels intact
    return expanded


def rgb(red: int, green: int, blue: int) -> Tuple[float, float, float]:
    return red / 255, green / 255, blue / 255


if __name__ == "__main__":
    main()
