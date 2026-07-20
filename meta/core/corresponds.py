import sys
import logging
import numpy as np
import nibabel as nib
from collections import defaultdict
from tslearn.metrics import dtw_path

from meta.io.streamline import read_streamlines
from meta.transforms.tractogram import transform_tractogram
from meta.utils.tractogram import reorient_streamlines

from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import length, transform_streamlines

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)


def get_centroids(streamlines, n_points=500, threshold=np.inf, num_clusters=None, sort_by=None, n_centroids=None):
    """
    Get centroids from a set of streamlines.

    Parameters
    ----------
    streamlines : A list of streamlines.
    n_points : Number of points to use for resampling the centroids.
    threshold : Distance threshold for clustering.
    num_clusters : Maximum number of QuickBundles clusters to create.
    sort_by : Sorting method for centroids. One of None, 'length', or 'cluster_size'.
    n_centroids : Number of centroids to return.

    Returns
    -------
    centroids : A list of centroids.
    """
    if sort_by not in (None, "length", "cluster_size"):
        raise ValueError("sort_by must be None, 'length', or 'cluster_size'.")

    feature = ResampleFeature(nb_points=n_points)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(threshold=threshold, metric=metric, max_nb_clusters=num_clusters)
    clusters = qb.cluster(streamlines)
    centroids = list(clusters.centroids)

    if not centroids:
        raise ValueError("QuickBundles did not create any centroids.")

    if sort_by == "length":
        order = np.argsort([length(c) for c in centroids])[::-1]
        centroids = [centroids[i] for i in order]
    elif sort_by == "cluster_size":
        order = np.argsort([len(c) for c in clusters])[::-1]
        centroids = [centroids[i] for i in order]

    if n_centroids is not None:
        centroids = centroids[:n_centroids]

    logging.info(f"Average length of centroids: {np.mean([length(c) for c in centroids])}")
    return centroids


def min_distance_filter(corres_pts, min_spacing=0.5, min_sls=5, detect_pts=None):
    """
    Drop points that did not align using DTW.

    When DTW finds no model correspondence for a subject point, that point collapses onto its
    neighbour, so two consecutive points end up closer than min_spacing voxels. Removing them
    keeps these unmatched points from distorting the segmentation.

    Parameters
    ----------
    corres_pts : Correspondence sets to keep, each of shape (num_points, 3).
    min_spacing : Minimum allowed distance (in voxels) between consecutive points.
    min_sls : Streamline-count cutoff between the two modes:
        >= min_sls sets -> drop whole sets whose closest pair is below min_spacing.
        <  min_sls sets -> drop the unmatched points, keeping the last point of each run.
    detect_pts : Sets used only to measure bunching (defaults to corres_pts). Pass the
        pre-interpolation correspondence so detection matches it.

    Returns
    -------
    kept : The filtered correspondence sets.
    original_indices : Original indices of the kept points.
    """
    sls = [np.asarray(s) for s in corres_pts]
    detect_sls = sls if detect_pts is None else [np.asarray(s) for s in detect_pts]
    n = sls[0].shape[0]

    if len(sls) >= min_sls:
        gap = np.array([np.linalg.norm(np.diff(s, axis=0), axis=1).min() for s in detect_sls])
        keep = gap >= min_spacing
        if keep.sum() < min_sls:
            keep = np.zeros(len(sls), bool)
            keep[np.argsort(gap)[::-1][:min_sls]] = True
        return [s for s, k in zip(sls, keep) if k], np.arange(n)

    gap = np.linalg.norm(np.diff(np.stack(sls), axis=1), axis=2).mean(0)
    keep_idx = [i for i in range(n - 1) if gap[i] >= min_spacing] + [n - 1]
    return [s[keep_idx] for s in sls], np.array(keep_idx)


def get_alignment(model, subject, num_segments, mask_img, transform=None, inverse=False, warp=None,
                    warp_source="ants", warp_first=False, trim_endpoints=True, method="hyperplane", min_spacing=0.5):
    """
    Get the alignment between model and subject streamlines.

    Parameters
    ----------
    model: Path to the model bundle.
    subject: Path to the subject bundle.
    num_segments: Number of segments for alignment.
    mask_img: Path to the mask image.
    transform: Path to the transformation matrix.
    inverse: Whether to invert the transformation matrix.
    warp: Path to the non-linear warp field.
    warp_source: Warp convention. One of 'ants' or 'dsi_studio'.
    warp_first: Whether to apply the warp before the affine transform.
    trim_endpoints: Whether to trim endpoints outside the warp grid.
    method: Correspondence method. One of 'hyperplane' or 'centerline'.

    Returns
    -------
    out_pts: List of updated correspondences using DTW.
    original_indices: Indices of DTW points.

    """

    if method not in ("hyperplane", "centerline"):
        raise ValueError("method must be either 'hyperplane' or 'centerline'.")

    ## Reference image:
    ref_image = nib.load(mask_img)

    ## Transform model to subject space:
    if transform is not None or warp is not None:
        model_sls, _, _, _ = transform_tractogram(tractogram=model, transform=transform, warp=warp, reference=mask_img, inverse=inverse,
                    warp_first=warp_first, warp_source=warp_source, trim_endpoints=trim_endpoints, output=None)
    else:
        model_sls, _, _, _ = read_streamlines(model)
    model_sls = transform_streamlines(model_sls, np.linalg.inv(ref_image.affine))

    ## Subject bundle:
    subj_sls, _, _, _ = read_streamlines(subject)
    subj_sls = transform_streamlines(subj_sls, np.linalg.inv(ref_image.affine))

    if method == "hyperplane":
        model_ctr = get_centroids(model_sls, n_points=num_segments, threshold=2.0, sort_by="cluster_size", n_centroids=1)
        subj_ctr = get_centroids(subj_sls, n_points=500, threshold=10.0, sort_by="cluster_size", n_centroids=1)
        subj_ctrs = get_centroids(subj_sls, n_points=500, threshold=0.01, num_clusters=500)
    else:
        model_ctr = get_centroids(model_sls, n_points=num_segments, threshold=2.0, sort_by="length", n_centroids=1)
        subj_ctr = get_centroids(subj_sls, n_points=500, threshold=6.0, sort_by="length", n_centroids=1)
        subj_ctrs = get_centroids(subj_sls, n_points=500, threshold=6.0, num_clusters=500, sort_by="length", n_centroids=1)

    logging.info(f"Using correspondence method: {method}")
    logging.info("Model centroid length: {}".format(np.mean([length(sl) for sl in model_ctr])))

    ## Check if the subject centroids are flipped compared to the model:
    subj_ctr = reorient_streamlines(subj_ctr, model_ctr)
    subj_ctrs = reorient_streamlines(subj_ctrs, model_ctr)

    ## Compute the correspondence between model and subject centroids:
    dtw_pts = []
    for model_sl, subj_sl in zip(model_ctr, subj_ctr):
        dtw_pairs, similarity_score = dtw_path(model_sl, subj_sl)
        logging.info(f"DTW similarity score: {similarity_score}")

        model_to_subj = defaultdict(list)
        for mi, si in dtw_pairs:
            model_to_subj[mi].append(si)

        kept_pairs = set()
        multi_idx = {mi for mi, si in model_to_subj.items() if len(si) > 1}

        if multi_idx:
            first_real = min(multi_idx)
            last_real = max(multi_idx)
            for mi in range(first_real, last_real + 1):
                si = model_to_subj[mi]
                kept_pairs.add((mi, si[len(si) // 2]))
        else:
            for mi, si in model_to_subj.items():
                kept_pairs.add((mi, si[len(si) // 2]))

        if multi_idx:
            first_real = min(multi_idx)
            last_real = max(multi_idx)
            for mi in range(first_real, last_real + 1):
                si = model_to_subj[mi]
                kept_pairs.add((mi, si[len(si) // 2]))
            for mi, si in model_to_subj.items():
                if mi < first_real or mi > last_real:
                    kept_pairs.add((mi, si[len(si) // 2]))
        else:
            for mi, si in model_to_subj.items():
                kept_pairs.add((mi, si[len(si) // 2]))

        ref_full = np.full((num_segments, 3), np.nan, dtype=float)
        for mi, si in kept_pairs:
            ref_full[mi] = subj_sl[si]
        dtw_pts.append(ref_full)

    ref_full = dtw_pts[0]
    valid_idx = np.where(~np.isnan(ref_full).all(axis=1))[0]
    n_valid = len(valid_idx)

    if n_valid == 0:
        raise ValueError("DTW correspondence did not keep any model positions.")

    ref_pts = ref_full[valid_idx]

    logging.info(f"Dynamic time warping (DTW) Correspondence shape: {ref_full.shape}")
    logging.info(f"Valid DTW indices: {valid_idx}")
    logging.info(f"Number of valid DTW anchors: {n_valid}")

    def expand_to_model(corr_pts, orig_idx, n_model):
        full = np.full((n_model, 3), np.nan, dtype=float)
        for k, idx in enumerate(orig_idx):
            if k < len(corr_pts):
                full[idx] = corr_pts[k]

        for dim in range(3):
            known = ~np.isnan(full[:, dim])
            kx = np.where(known)[0]
            ky = full[known, dim]
            if len(kx) >= 2:
                full[:, dim] = np.interp(np.arange(n_model), kx, ky)
            elif len(kx) == 1:
                full[:, dim] = ky[0]
        return full

    subj_pts = []
    for subj_sl in subj_ctrs:
        subj_sl = np.squeeze(subj_sl)
        dtw_pairs, similarity_score = dtw_path(ref_pts, subj_sl)
        # logging.info(f"Subject centroid DTW similarity score: {similarity_score}")
        ref_to_ctr = defaultdict(list)
        for ref_i, ctr_i in dtw_pairs:
            ref_to_ctr[ref_i].append(ctr_i)

        ctr_pts = []
        for ref_i in range(n_valid):
            ctr_idx = ref_to_ctr[ref_i]
            ctr_pts.append(subj_sl[ctr_idx[len(ctr_idx) // 2]])

        full_ctr = expand_to_model(ctr_pts, valid_idx, num_segments)
        subj_pts.append(full_ctr)

    ref_expanded = expand_to_model(ref_pts, valid_idx, num_segments)
    all_pts = np.stack([ref_expanded] + subj_pts, axis=0)
    logging.info(f"Combined Correspondence shape: {all_pts.shape}")

    lengths = [length(sl) for sl in all_pts]
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    length_threshold = mean_length - (3 * std_length)
    logging.info(f"Average streamlines length: {mean_length:.3f}, std: {std_length:.3f}")

    short_idx = np.where(np.array(lengths) < length_threshold)[0]
    filt_pts = [sl for idx, sl in enumerate(all_pts) if idx not in short_idx]
    logging.info(f"Final Correspondence shape: {np.array(filt_pts).shape}")

    ## Compute pairwise distances:
    corr_pts = np.array(filt_pts)
    pair_dists = np.zeros((corr_pts.shape[1], corr_pts.shape[0], corr_pts.shape[0]))
    for i in range(corr_pts.shape[1]):
        for j in range(corr_pts.shape[0]):
            for k in range(j + 1, corr_pts.shape[0]):
                pair_dists[i, j, k] = np.linalg.norm(corr_pts[j, i] - corr_pts[k, i])

    n_streamlines = corr_pts.shape[0]
    lower_tri_mask = np.tril(np.ones((n_streamlines, n_streamlines), dtype=bool))
    pair_dists[:, lower_tri_mask] = np.nan

    std_dists = np.nanstd(pair_dists, axis=(1, 2))
    core_idx = np.where(std_dists <= 5)[0]

    if core_idx.size == 0 and not np.all(np.isnan(std_dists)):
        median_std = np.nanmedian(std_dists)
        logging.warning("Fixed std threshold (5 vox) found no compact positions; "
            f"switching to adaptive median threshold ({median_std:.3f} vox).")
        core_idx = np.where(std_dists <= median_std)[0]

    logging.info(f"Core indices based on std distances: {core_idx}")

    if core_idx.size > 0:
        core_start = core_idx[0]
        core_end = core_idx[-1]
        out_pts = []

        for array in filt_pts:
            merged = []
            if core_start > 1:
                start_pt = array[0]
                end_pt = array[core_start]
                side_1_pts = np.linspace(start_pt, end_pt, core_start + 1)[1:-1]
                merged.extend(array[0:1])
                merged.extend(side_1_pts)
            else:
                merged.extend(array[0:core_start])

            merged.extend(array[core_start:core_end + 1])
            if num_segments - core_end > 1:
                start_pt = array[core_end]
                end_pt = array[-1]
                side_2_pts = np.linspace(start_pt, end_pt, num_segments - core_end)[1:-1]
                merged.extend(side_2_pts)
                merged.extend(array[-1:])

            out_pts.append(np.array(merged))
    else:
        out_pts = filt_pts

    logging.info(f"Shape of updated correspondences: {np.array(out_pts).shape}")

    return min_distance_filter(out_pts, min_spacing=min_spacing, detect_pts=filt_pts)
