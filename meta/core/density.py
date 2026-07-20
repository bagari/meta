import sys
import logging
import argparse
import numpy as np
import nibabel as nib
from dipy.tracking import utils

from meta.io.streamline import read_streamlines

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)


def density_map():
    parser = argparse.ArgumentParser(description='Convert streamlines of white matter bundle into a density map and binary mask.')
    parser.add_argument('--tractogram', type=str, help='Path to the bundle file containing streamlines', required=True)
    parser.add_argument('--reference', type=str, help='Path to the reference image', required=True)
    parser.add_argument('--output', type=str, help='Path to the output binary mask file', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    # Load reference image
    ref_img = nib.load(args.reference)
    ref_affine = ref_img.affine
    ref_shape = ref_img.get_fdata().shape

    # Load streamlines with ref for tck
    streamlines, _, _, _ = read_streamlines(args.tractogram, reference=args.reference)
    max_seq_len = np.linalg.norm(ref_affine[:3, :3], axis=0).min() / 4
    streamlines = list(utils.subsegment(streamlines, max_seq_len))

    # Filter streamline points outside the reference FOV.
    inv_aff = np.linalg.inv(ref_affine)
    dim = np.asarray(ref_shape[:3], dtype=np.float32)
    total_pts = sum(len(s) for s in streamlines)
    filtered, dropped_pts, dropped_streamlines = [], 0, 0
    for s in streamlines:
        s = np.asarray(s)
        vox = s @ inv_aff[:3, :3].T + inv_aff[:3, 3]
        inside = np.all((vox >= 0) & (vox <= dim - 1), axis=1)
        n_in = int(np.count_nonzero(inside))
        if n_in == len(s):
            filtered.append(s)
        elif n_in >= 2:
            filtered.append(s[inside])
            dropped_pts += len(s) - n_in
        else:
            dropped_pts += len(s)
            dropped_streamlines += 1
    streamlines = filtered

    if dropped_pts or dropped_streamlines:
        pct = 100.0 * dropped_pts / max(total_pts, 1)
        logging.info(f'FOV filter: dropped {dropped_pts}/{total_pts} points ({pct:.3f}%); '
                    f'{dropped_streamlines} streamlines lost (<2 in-grid points remained).')

    if not streamlines:
        logging.error('No streamlines remain after FOV filtering — cannot compute density map.')
        sys.exit(1)

    # Create Density Map
    density_map = utils.density_map(streamlines, vol_dims=ref_shape, affine=ref_affine)

    ## Convert Density Map to Binary Mask
    mask = density_map > 0
    nib.save(nib.Nifti1Image(mask.astype("uint8"), ref_affine), args.output)
    logging.info(f'Binary mask saved to {args.output}')

    ## save the density map
    density_img = nib.Nifti1Image(density_map.astype("float32"), ref_affine)
    nib.save(density_img, args.output.replace('.nii.gz', '_density.nii.gz'))
    logging.info(f'Density map and binary mask saved to {args.output.replace(".nii.gz", "_density.nii.gz")}')
