import sys
import logging
import argparse
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform, inv_ornt_aff

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.streamline import deform_streamlines, transform_streamlines

from meta.io.streamline import read_streamlines
from meta.io.transform import load_transformation

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)


def apply_affine(streamlines, aff, inverse=False):
    """Apply a 4x4 affine transform to streamlines in RAS mm space."""
    if aff.shape != (4, 4):
        raise ValueError("Affine transform matrix must be 4x4.")
    if inverse:
        aff = np.linalg.inv(aff)
    return list(transform_streamlines(streamlines, aff))

def trim_endpoints_to_grid(streamlines, grid_affine, grid_shape):
    """Trim only leading/trailing points outside a voxel grid."""
    inv_aff = np.linalg.inv(grid_affine)
    grid_shape = np.asarray(grid_shape, dtype=np.float32)
    trimmed = []
    n_points = 0
    n_streamlines = 0
    n_inside_gaps = 0
    for s in streamlines:
        vox = s @ inv_aff[:3, :3].T + inv_aff[:3, 3]
        inside = np.all((vox >= 0) & (vox <= grid_shape - 1), axis=1)
        if inside.all():
            trimmed.append(s)
            continue
        start = 0
        end = len(s)
        while start < end and not inside[start]:
            start += 1
        while end > start and not inside[end - 1]:
            end -= 1
        n_points += start + len(s) - end
        n_inside_gaps += np.count_nonzero(~inside[start:end])
        if end - start >= 2:
            trimmed.append(s[start:end])
        else:
            n_streamlines += 1
    return trimmed, n_points, n_streamlines, n_inside_gaps

def load_warp_field(warp_path, source="ants", ref_affine=None, ref_shape=None):
    """Load a warp field and return a RAS-mm displacement field."""
    warp_img = nib.load(warp_path)
    warp_data = warp_img.get_fdata(dtype=np.float32)

    if warp_data.ndim == 5:
        if warp_data.shape[3] != 1:
            raise ValueError(f"5D warp field must have singleton 4th dimension, got {warp_data.shape}.")
        warp_data = warp_data[:, :, :, 0, :]

    if warp_data.ndim != 4 or warp_data.shape[-1] != 3:
        raise ValueError(f"Warp field must have shape (X, Y, Z, 3), got {warp_data.shape}.")

    if source == "ants":
        warp_data = warp_data.copy()
        warp_data[..., :2] *= -1

    elif source == "dsi_studio":
        if ref_affine is None:
            raise ValueError("warp_source='dsi_studio' requires a target-space --reference.")

        src_ornt = io_orientation(ref_affine)
        dst_ornt = io_orientation(warp_img.affine)
        if not np.array_equal(src_ornt, dst_ornt):
            if ref_shape is None:
                raise ValueError("ref_shape is required for DSI-Studio orientation.")
            ornt = ornt_transform(src_ornt, dst_ornt)
            ref_affine = ref_affine @ inv_ornt_aff(ornt, ref_shape)
        shape = warp_data.shape[:3]
        ijk = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0)
        src_aff = warp_img.affine.astype(np.float32)
        ref_aff = ref_affine.astype(np.float32)
        src_ras = ijk @ src_aff[:3, :3].T + src_aff[:3, 3]
        ref_ras = warp_data @ ref_aff[:3, :3].T + ref_aff[:3, 3]
        warp_data = (ref_ras - src_ras).astype(np.float32)
    else:
        raise ValueError(f"Unknown warp source: {source!r}. Supported sources are 'ants' and 'dsi_studio'.")

    return warp_img, warp_data

def apply_nonlinear_warp(streamlines, warp_data, warp_affine, ref_affine=None, trim_endpoints=True):
    """Apply a non-linear warp field to streamlines in RAS mm space."""
    if ref_affine is None:
        ref_affine = warp_affine
    if trim_endpoints:
        streamlines, n_points, n_streamlines, n_inside_gaps = trim_endpoints_to_grid(streamlines, warp_affine, warp_data.shape[:3])
        logging.info(f"Trimmed {n_points} endpoint points outside the warp grid.")
        if n_streamlines:
            logging.info(f"Dropped {n_streamlines} streamlines with fewer than 2 valid points after trimming.")
        if n_inside_gaps:
            logging.info(f"[info] {n_inside_gaps} interior points are outside the warp grid and were kept.")
    inv_warp_aff = np.linalg.inv(warp_affine)
    inv_ref_aff = np.linalg.inv(ref_affine)
    warped = deform_streamlines(streamlines, deform_field=warp_data, stream_to_current_grid=inv_warp_aff, current_grid_to_world=warp_affine, stream_to_ref_grid=inv_ref_aff, ref_grid_to_world=ref_affine)

    return list(warped)


def transform_tractogram(tractogram, transform=None, warp=None, reference=None, inverse=False, warp_first=False, warp_source="ants", trim_endpoints=True, output=None):
    """Load streamlines, apply affine and/or warp, then optionally save."""

    if transform is None and warp is None:
        raise ValueError("Provide transform, warp, or both.")

    if output is not None and reference is None:
        raise ValueError("reference is required to save the output tractogram.")

    use_affine = transform is not None
    if warp_source == "dsi_studio" and warp is not None:
        if use_affine:
            logging.info("DSI-Studio warp includes the affine transform; skipping --transform.")
        use_affine = False

    if inverse and not use_affine:
        logging.info("--inverse ignored because no affine transform is applied.")
    logging.info(f"Loading tractogram: {tractogram}")
    streamlines, groups, _, _ = read_streamlines(tractogram, reference=reference)
    logging.info(f"{len(streamlines)} streamlines loaded.")

    if use_affine:
        logging.info(f"Loading affine transform: {transform}")
        aff = load_transformation(transform)
        logging.info(f"Transform matrix:\n{aff}")
    else:
        logging.info("No affine transform will be applied.")
        aff = None

    warp_data = None
    warp_affine = None
    ref_affine = None
    ref_shape = None
    if warp is not None:
        logging.info(f"Loading warp field: {warp}")
        if reference is not None:
            ref_img = nib.load(reference)
            ref_affine = ref_img.affine
            ref_shape = ref_img.shape[:3]

        warp_img, warp_data = load_warp_field(warp, source=warp_source, ref_affine=ref_affine, ref_shape=ref_shape)
        warp_affine = warp_img.affine
        if ref_affine is None:
            ref_affine = warp_affine
            ref_shape = warp_data.shape[:3]
        logging.info(f"Warp field shape: {warp_data.shape}")
    else:
        logging.info("No warp field provided; skipping non-linear step.")

    logging.info("Applying transforms ...")

    if aff is not None and warp_data is not None:
        if warp_first:
            logging.info("[warp -> affine (linear)]")
            streamlines = apply_nonlinear_warp(streamlines, warp_data, warp_affine, ref_affine, trim_endpoints)
            streamlines = apply_affine(streamlines, aff, inverse)
        else:
            logging.info("[affine (linear) -> warp]")
            streamlines = apply_affine(streamlines, aff, inverse)
            streamlines = apply_nonlinear_warp(streamlines, warp_data, warp_affine, ref_affine, trim_endpoints)

    elif aff is not None:
        logging.info("[affine (linear) only]")
        streamlines = apply_affine(streamlines, aff, inverse)

    elif warp_data is not None:
        logging.info("[warp only]")
        streamlines = apply_nonlinear_warp(streamlines, warp_data, warp_affine, ref_affine, trim_endpoints)
    logging.info(f"Done. {len(streamlines)} streamlines after transform.")

    if output is not None:
        logging.info(f"Saving to: {output}")
        ref_img = nib.load(reference)
        sft = StatefulTractogram(streamlines, reference=ref_img, space=Space.RASMM)
        save_tractogram(sft, output, bbox_valid_check=False)
        logging.info("Saved successfully.")

    return streamlines, groups, ref_affine, ref_shape


def apply_transform_tractogram():
    parser = argparse.ArgumentParser(description="Apply an affine transformation and/or a non-linear warp field to the streamlines of a tractogram.")

    parser.add_argument("--tractogram", required=True, help="Input tractogram (.trk / .tck / .tt.gz).")
    parser.add_argument("--transform", default=None, help="Linear/affine transform file, supported (.mat, .txt, .npy, .mz).")
    parser.add_argument("--warp", default=None, help="Non-linear warp/mapping NIfTI.")
    parser.add_argument("--output", required=True, help="Output tractogram (.trk / .tck).")
    parser.add_argument("--reference", required=True, help="Target-space reference image in NIfTI format (.nii / .nii.gz).")
    parser.add_argument("--warp_source", choices=["ants", "dsi_studio"], default="ants", help="Warp convention: ants or dsi_studio.")
    parser.add_argument("--inverse", action="store_true", help="Apply the inverse affine (linear) transform. This does not invert the warp.")
    parser.add_argument("--warp_first", action="store_true", help="Apply the warp before the affine transform. Default is affine first, then warp.")
    parser.add_argument("--no_trim", action="store_true", help="Keep endpoint points outside the warp grid instead of trimming them.")

    args = parser.parse_args()

    if args.transform is None and args.warp is None:
        parser.error("provide --transform, --warp, or both.")

    transform_tractogram(tractogram=args.tractogram, transform=args.transform, warp=args.warp, reference=args.reference, inverse=args.inverse, warp_first=args.warp_first, warp_source=args.warp_source, trim_endpoints=not args.no_trim, output=args.output)


if __name__ == "__main__":
    apply_transform_tractogram()
