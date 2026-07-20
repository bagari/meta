import numpy as np
from tqdm import tqdm


def segment_bundle(bundle_data, corres_pts, num_segments):
    """
    Parcellate white matter bundle into specified segments based on correspondence points.

    Parameters:
    -----------
    bundle_data: A binary mask of the white matter bundle as a NumPy array
    corres_pts: A list of arrays with shape (num_segments, 3) containing correspondence points.
    num_segments: The required number of segments to divide the bundle into

    Returns:
    --------
    segments: A list of labels, where each label corresponds to a segment.
    """

    bundle_mask = bundle_data.astype(bool)
    segments = [np.zeros_like(bundle_mask, dtype=bool) for _ in range(num_segments + 1)]
    vox_coords = np.argwhere(bundle_mask)

    for seg_pts in tqdm(corres_pts):
        for i in range(num_segments):
            if i == 0:
                plane_norm = (seg_pts[i + 1] - seg_pts[i]).astype(float)
                mask = (vox_coords - seg_pts[i]) @ (-plane_norm) >= 0
                segments[i][vox_coords[mask, 0], vox_coords[mask, 1], vox_coords[mask, 2]] = True

            if 0 <= i < num_segments - 2:
                plane_norm = (seg_pts[i + 1] - seg_pts[i]).astype(float)
                next_norm = (seg_pts[i + 2] - seg_pts[i + 1]).astype(float)
                mask = (((vox_coords - seg_pts[i]) @ plane_norm >= 0) & ((vox_coords - seg_pts[i + 1]) @ (-next_norm) >= 0))
                segments[i + 1][vox_coords[mask, 0], vox_coords[mask, 1], vox_coords[mask, 2]] = True

            elif i == num_segments - 2: 
                plane_norm = (seg_pts[i] - seg_pts[i - 1]).astype(float)
                mask = (vox_coords - seg_pts[i - 1]) @ plane_norm >= 0
                segments[i + 1][vox_coords[mask, 0], vox_coords[mask, 1], vox_coords[mask, 2]] = True

            elif i == num_segments - 1:
                plane_norm = (seg_pts[i] - seg_pts[i - 1]).astype(float)
                mask = (vox_coords - seg_pts[i]) @ plane_norm >= 0
                segments[i + 1][vox_coords[mask, 0], vox_coords[mask, 1], vox_coords[mask, 2]] = True

    seg_arr = np.array(segments)
    seg_sum = np.sum(seg_arr, axis=0)

    all_pts = np.vstack(corres_pts)
    seg_idx = np.tile(np.arange(num_segments), len(corres_pts))

    def assign_to_nearest(coords, chunk_size=25000):
        """Assign each voxel coordinate to its nearest correspondence point."""
        for start in range(0, len(coords), chunk_size):
            chunk = coords[start:start + chunk_size]
            diff = chunk[:, np.newaxis, :] - all_pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            closest = seg_idx[np.argmin(dists, axis=1)]

            for idx in range(num_segments):
                winners = chunk[closest == idx]
                if len(winners) > 0:
                    segments[idx][winners[:, 0], winners[:, 1], winners[:, 2]] = True

    conflict_coords = np.argwhere(seg_sum >= 2)
    if len(conflict_coords) > 0:
        for seg in segments:
            seg[conflict_coords[:, 0], conflict_coords[:, 1], conflict_coords[:, 2]] = False
        assign_to_nearest(conflict_coords)

    missing_coords = np.argwhere((seg_sum == 0) & bundle_mask)
    if len(missing_coords) > 0:
        assign_to_nearest(missing_coords)

    return segments

