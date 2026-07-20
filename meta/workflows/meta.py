import os
import sys
import logging
import argparse
import numpy as np
import nibabel as nib

from meta.core.mesh import medial_core
from meta.core.segments import segment_bundle
from meta.core.corresponds import get_alignment
from meta.core.medial_surface import generate_medial_surface

import warnings
warnings.simplefilter("ignore")

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)


def main():
    print("meta-neuro (MeTA) | Iyad Ba Gari | https://github.com/bagari/meta", file=sys.stderr)
    parser = argparse.ArgumentParser(
        description='Medial Tractography Analysis (MeTA) for White Matter Bundle Parcellation. '
                    'Author: Iyad Ba Gari <iyad.bagari@usc.edu>. '
                    'Repository: https://github.com/bagari/meta')
    parser.add_argument("--subject", type=str, help='Subject IDs')
    parser.add_argument("--bundle", type=str, help='Name of white matter bundle')
    
    parser.add_argument("--mbundle", type=str, help='Streamlines of model bundle', required=True)
    parser.add_argument("--sbundle", type=str, help='Streamlines of subject bundle', required=True)
    parser.add_argument("--mask", type=str, help='Mask of subject white matter bundle', required=True)
    parser.add_argument("--percent", type=float, help='Percent of distance to keep from each side of Medial surface' , default=0.125)
    parser.add_argument("--save_medial", action='store_true', help='Save the boundary mesh and medial surface (.vtk) to the output dir')
    parser.add_argument("--level", type=float, default=0.1, help='Iso-surface level for the boundary mesh (marching cubes)')
    parser.add_argument("--prune", type=float, default=1.5, help='Geodesic/Euclidean pruning factor for the medial surface')

    parser.add_argument("--transform", type=str, help='Transformation matrix: MNI → subject space', required=False, default=None)
    parser.add_argument("--inverse", action='store_true', help='Inverse transformation matrix if direction subject → MNI')
    parser.add_argument("--warp", type=str, help='Non-linear warp field/mapping applied to the model bundle', required=False, default=None)
    parser.add_argument("--warp_source", type=str, choices=("ants", "dsi_studio"), help='Warp convention: ants or dsi_studio', default="dsi_studio")
    parser.add_argument("--warp_first", action='store_true', help='Apply the warp before the affine transform')
    parser.add_argument("--no_trim", action='store_true', help='Keep endpoints outside the warp grid')

    parser.add_argument("--num_segments", type=int, help='The required number of segments along the bundle length', default=15)
    parser.add_argument("--seg_method", type=str, choices=("hyperplane", "centerline"), help='Segmentation method for correspondence estimation', default="hyperplane")
    parser.add_argument("--output", type=str, help='Output directory' , required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load mask (global all bundle):
    img = nib.load(args.mask)
    mask_data = img.get_fdata()

    ## if used --save_medial, the medial surface and volume will be saved:
    logging.info(f'Building medial surface for {args.bundle} bundle ...')
    mesh_out = os.path.join(args.output, args.subject + '_' + args.bundle + ".vtk") if args.save_medial else None
    skel_out = os.path.join(args.output, args.subject + '_' + args.bundle + "_medial_surface.vtk") if args.save_medial else None
    volume, medial = generate_medial_surface(args.mask, mesh_out, skel_out, threshold=args.level, prune=args.prune)

    # Extract medial core volume:
    logging.info(f'Computing global core for {args.bundle} bundle ...')
    core, _ = medial_core(bundle_mask = args.mask, medial_surface = medial, volume_mesh = volume, percent = args.percent)
    nib.save(core, os.path.join(args.output, args.subject + '_' + args.bundle + "_global_core.nii.gz"))

    ## Get corresponding points between model and subject bundle:
    logging.info(f'Getting corresponding points between model and subject bundle...')
    corres_points, original_indices = get_alignment(
        model = args.mbundle,
        subject = args.sbundle,
        num_segments = args.num_segments,
        mask_img = args.mask,
        transform = args.transform,
        inverse = args.inverse,
        warp = args.warp,
        warp_source = args.warp_source,
        warp_first = args.warp_first,
        trim_endpoints = not args.no_trim,
        method = args.seg_method,
    )

    ## Segment the bundle using DTW points:
    logging.info(f'Segmenting {args.bundle} into {args.num_segments} segments')
    segments = segment_bundle(bundle_data = mask_data, corres_pts = corres_points, num_segments = len(original_indices))

    segmented_bundle = np.zeros(mask_data.shape)
    for new_i, orig_i in enumerate(original_indices):
        segmented_bundle[segments[new_i]] = orig_i + 1

    logging.info('Saving segmentation along length results...')
    # Local all segments:
    local_all = nib.Nifti1Image(segmented_bundle, affine=img.affine)
    nib.save(local_all, os.path.join(args.output, args.subject + '_' + args.bundle + '_' + str(args.num_segments) + "_segments_local_all.nii.gz"))

    core_bundle = core.get_fdata()
    core_indices = np.where(core_bundle > 0)
    core_bundle[core_indices] = segmented_bundle[core_indices]

    # Local core segments:
    local_core = nib.Nifti1Image(core_bundle, affine=img.affine)
    nib.save(local_core, os.path.join(args.output, args.subject + '_' + args.bundle + '_' + str(args.num_segments) + "_segments_local_core.nii.gz"))

if __name__ == '__main__':
    main()
