# We follow the evaluation method of CellPose, so on the X-axis IoU from 0.5 to 1, and on the y axis average precision.

import os
from typing import Tuple, List, Optional, Iterable, Dict

import numpy
import skimage.measure
import tifffile
from matplotlib import pyplot as plt
from numpy import ndarray

import figure_lib

_GROUND_TRUTH_FOLDER = r"P:\Rodriguez_Colman\vidi_rodriguez_colman\rkok\data_analysis\2023-05 RK0029 Rutger Measuring CellPose performance\Manual segmentation"
_AUTOMATIC_FOLDER = r"P:\Rodriguez_Colman\vidi_rodriguez_colman\rkok\data_analysis\2023-05 RK0029 Rutger Measuring CellPose performance\CellPose segmentation"


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
                                  ) -> Tuple[float, Optional[int]]:
        """Gets the intersection-over-union of the automatic nucleus with the corresponding ground truth nucleus. If
        the automatic nucleus is a false positive, then 0 will be returned. If it overlaps with multiple nuclei in the
        ground truth, then the highest IoU is returned.

        Returns the intersection of union, as well as the ground truth nucleus that it intersected with."""
        max_iou = 0
        with_label = None
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
        return max_iou, with_label

    def get_nucleus_ids_and_z(self) -> Iterable[Tuple[int, float]]:
        """Gets all nucleus ids that are in the ground truth set."""
        for regionprops in self._regionprops:
            if regionprops is not None:
                yield regionprops.label, regionprops.centroid[0]


class _ImageResults:
    """Given intersection of unions and a list of ground truth nuclei, this class calculates the true positives, false
    positives and false negatives."""
    _ground_truth_nuclei_to_z: Dict[int, float]
    _overlaps: List[
        Tuple[Optional[int], int, float, float]]  # Ground truth nucleus id, predicted nucleus id, intersection over union

    def __init__(self, ground_truth_nuclei_to_z: Dict[int, float]):
        self._ground_truth_nuclei_to_z = ground_truth_nuclei_to_z
        self._overlaps = list()

    def add_overlap(self, ground_truth_nucleus: Optional[int], predicted_nucleus: int, intersection_over_union: float,
                    *, z: float):
        """Registers the overlap between a nucleus in the ground truth and a predicted nucleus.

        If a predicted nucleus had no overlap with any ground truth nucleus, set the ground truth nucleus to None and
        use 0 for the intersection_over_union.
        """
        self._overlaps.append((ground_truth_nucleus, predicted_nucleus, intersection_over_union, z))

    def true_positives(self, *, iou_cutoff: float = 0.5, z_cutoff: float = float("inf")) -> int:
        true_positives = 0
        for overlap in self._overlaps:
            if overlap[2] >= iou_cutoff and overlap[3] < z_cutoff:
                true_positives += 1
        return true_positives

    def false_negatives(self, *, iou_cutoff: float = 0.5, z_cutoff: float = float("inf")) -> int:
        # Find nuclei within z range
        unused_nuclei = set()
        for unused_nucleus, z in self._ground_truth_nuclei_to_z.items():
            if z < z_cutoff:
                unused_nuclei.add(unused_nucleus)

        # Check which nuclei are actually used
        for overlap in self._overlaps:
            if overlap[0] in unused_nuclei and overlap[2] >= iou_cutoff and overlap[3] < z_cutoff:
                unused_nuclei.remove(overlap[0])

        # What remains are nuclei that weren't detected
        return len(unused_nuclei)

    def false_positives(self, *, iou_cutoff: float = 0.5, z_cutoff: float = float("inf")) -> int:
        false_positives = 0
        for overlap in self._overlaps:
            if overlap[2] < iou_cutoff and overlap[3] < z_cutoff:
                false_positives += 1
        return false_positives


class _OverallResults:
    """Calculates the overall scores based on all the images."""
    _results: List[_ImageResults]

    def __init__(self):
        self._results = list()

    def add_results(self, results: _ImageResults):
        self._results.append(results)

    def precision(self, *, iou_cutoff: float, z_cutoff: float = float("inf")) -> float:
        true_positives = sum([result.true_positives(iou_cutoff=iou_cutoff, z_cutoff=z_cutoff) for result in self._results])
        false_positives = sum([result.false_positives(iou_cutoff=iou_cutoff, z_cutoff=z_cutoff) for result in self._results])
        return true_positives / (true_positives + false_positives)

    def recall(self, *, iou_cutoff: float, z_cutoff: float = float("inf")) -> float:
        true_positives = sum([result.true_positives(iou_cutoff=iou_cutoff, z_cutoff=z_cutoff) for result in self._results])
        false_negatives = sum([result.false_negatives(iou_cutoff=iou_cutoff, z_cutoff=z_cutoff) for result in self._results])
        return true_positives / (true_positives + false_negatives)

    def f1_score(self, *, iou_cutoff: float, z_cutoff: float = float("inf")) -> float:
        precision = self.precision(iou_cutoff=iou_cutoff, z_cutoff=z_cutoff)
        recall = self.recall(iou_cutoff=iou_cutoff, z_cutoff=z_cutoff)
        return 2 * (precision * recall) / (precision + recall)


def _place_at(into: ndarray, start_zyx: Tuple[int, int, int], image: ndarray):
    into[start_zyx[0]:start_zyx[0] + image.shape[0],
    start_zyx[1]:start_zyx[1] + image.shape[1],
    start_zyx[2]:start_zyx[2] + image.shape[2]] = image


def _handle_image(ground_truth: ndarray, automatic: ndarray) -> _ImageResults:
    """Calculates the intersection over union for all nuclei, so that you can later on calculate the true/false
     positives and false negatives."""
    ground_truth_masks = _GroundTruthImage(ground_truth)
    results = _ImageResults(dict(ground_truth_masks.get_nucleus_ids_and_z()))

    for automatic_nucleus in skimage.measure.regionprops(automatic):
        iou, nucleus_id = ground_truth_masks.get_intersection_of_union(automatic_nucleus)
        z = (automatic_nucleus.bbox[0] + automatic_nucleus.bbox[3]) / 2
        results.add_overlap(nucleus_id, automatic_nucleus.label, iou, z=z)

    return results


def main():
    results = _OverallResults()
    for file in os.listdir(_GROUND_TRUTH_FOLDER):
        if not file.endswith(".tif"):
            continue

        ground_truth = tifffile.imread(os.path.join(_GROUND_TRUTH_FOLDER, file))
        automatic = tifffile.imread(os.path.join(_AUTOMATIC_FOLDER, file))
        results.add_results(_handle_image(ground_truth, automatic))

    _plot_by_z(results)
    _plot_by_ioc_cutoff(results)


def _plot_by_z(results: _OverallResults):
    z_values_px = numpy.linspace(5, 30)
    precisions = [results.precision(iou_cutoff=0.5, z_cutoff=z_px) for z_px in z_values_px]
    recalls = [results.recall(iou_cutoff=0.5, z_cutoff=z_px) for z_px in z_values_px]
    figure = figure_lib.new_figure()
    ax = figure.gca()
    ax.plot(z_values_px, precisions, label="Precision")
    ax.plot(z_values_px, recalls, label="Recall")
    ax.set_xlabel("Z (px)")
    ax.legend()
    plt.show()


def _plot_by_ioc_cutoff(results: _OverallResults):
    iou_cutoffs = numpy.linspace(0, 1)
    precisions = [results.precision(iou_cutoff=iou_cutoff) for iou_cutoff in iou_cutoffs]
    recalls = [results.recall(iou_cutoff=iou_cutoff) for iou_cutoff in iou_cutoffs]
    figure = figure_lib.new_figure()
    ax = figure.gca()
    ax.plot(iou_cutoffs, precisions, label="Precision")
    ax.plot(iou_cutoffs, recalls, label="Recall")
    ax.set_xlabel("IoU matching threshold")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
