import os
import re
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import map_coordinates
from meta.io.streamline import read_streamlines
from dipy.tracking.streamline import transform_streamlines

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)

def named_paths(values):
    # name=path pairs; a plain path is named after its file, without .nii or .nii.gz.
    return dict(v.split('=', 1) if '=' in v else (re.sub(r'\.nii(\.gz)?$', '', os.path.basename(v)), v) for v in values)


def segment_profile(micro_map, labels, subject, bundle, voxelwise=False):
    # Mean of the non-zero map values in each mask label, and optionally every voxel value.
    means = {'segment': [], 'mean': []}
    voxels = {'segment': [], 'value': []}
    for label in np.unique(labels):
        label = int(label)
        if label == 0:
            continue
        in_label = labels == label
        means['segment'].append(label)
        means['mean'].append(np.nanmean(micro_map[in_label & (micro_map != 0)]))
        if voxelwise:
            voxels['segment'].extend([label] * int(in_label.sum()))
            voxels['value'].extend(micro_map[in_label])
    return (pd.DataFrame({'subjectID': subject, 'bundle': bundle, **means}),
            pd.DataFrame({'subjectID': subject, 'bundle': bundle, **voxels}))


## Compute bundle features based on binary masks and microstructure maps
def volumetric_profile():
    parser = argparse.ArgumentParser(description='Compute volumetric profiles (segment means and, optionally, voxel values) of a white matter bundle for one or more maps and masks.')
    parser.add_argument('--subject', type=str, help='Subject ID', required=True)
    parser.add_argument('--bundle', type=str, help='White matter bundle name', required=True)
    parser.add_argument('--mask', type=str, nargs='+', help='White matter bundle mask(s) or label image(s), as path or name=path, e.g. local_all=CST_local_all.nii.gz', required=True)
    parser.add_argument('--map', type=str, nargs='+', help='Brain microstructure map(s), as path or name=path, e.g. fa=subject_FA.nii.gz', required=True)
    parser.add_argument('--output', type=str, help='Output directory to save extracted features', required=True)
    parser.add_argument('--voxelwise', action='store_true', help='Save voxel-wise in HDF5 format')

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    maps, masks = named_paths(args.map), named_paths(args.mask)
    if len(maps) < len(args.map) or len(masks) < len(args.mask):
        parser.error('--map and --mask names must be unique.')

    # One unnamed map and mask keep the original file names; otherwise the map and mask names are added.
    single = len(args.map) == len(args.mask) == 1 and '=' not in args.map[0] + args.mask[0]
    def output_path(map_name, mask_name, kind, ext):
        name = f'segments_{kind}' if single else f'{map_name}_segments_{kind}_{mask_name}'
        return f'{args.output}/{args.subject}_{args.bundle}_{name}.{ext}'

    # Load each mask and map once; a failed map is reported after the others are profiled.
    mask_data = {name: nib.load(path).get_fdata() for name, path in masks.items()}
    failed = []
    for map_name, map_path in maps.items():
        try:
            micro_map = nib.load(map_path).get_fdata()
            for mask_name, labels in mask_data.items():
                mean_df, voxel_df = segment_profile(micro_map, labels, args.subject, args.bundle, args.voxelwise)
                csv_path = output_path(map_name, mask_name, 'average', 'csv')
                mean_df.to_csv(csv_path, index=False)
                logging.info(f"Saved segment-wise mean features to {csv_path}")
                if args.voxelwise:
                    hdf5_path = output_path(map_name, mask_name, 'voxelwise', 'h5')
                    voxel_df.to_hdf(hdf5_path, key='df', mode='w', complevel=9)
                    logging.info(f"Saved voxel-wise features to {hdf5_path}")
        except Exception as e:
            logging.error(f'An error occurred while processing {map_path}: {str(e)}')
            failed.append(map_name)

    if failed:
        sys.exit(f"Failed to profile map(s): {', '.join(failed)}")


def streamlines_profile():
    parser = argparse.ArgumentParser(description='Compute streamlines profile (average mean and point-wise) of a white matter bundle.')
    parser.add_argument("--subject", type=str, help='Subject ID', required=True)
    parser.add_argument("--bundle", type=str, help='White matter bundle name', required=True)
    parser.add_argument("--tractogram", type=str, help='Streamlines of subject bundle', required=True)
    parser.add_argument("--mask", type=str, help='Path to white matter bundle mask', required=True)
    parser.add_argument("--map", type=str, help='Brain microstructure map, e.g. FA, MD, etc.', required=True)
    parser.add_argument("--output", type=str, help='Output directory to save extracted features', required=True)
    parser.add_argument("--pointwise", action='store_true', help='Save point-wise in HDF5 format')

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    try:
        ## metrics map and bundle mask/labels:
        micro_map = nib.load(args.map)
        metric_data = micro_map.get_fdata()

        labels_img = nib.load(args.mask)
        labels_data = labels_img.get_fdata()

        ## Load streamlines:
        streamlines, _, _, _ = read_streamlines(args.tractogram)
        print(f"Number of streamlines: {len(streamlines)}")
        sls_voxels = transform_streamlines(streamlines, np.linalg.inv(micro_map.affine))
        sls_pts = np.vstack(sls_voxels)

        ## Extract microstructure values and labels for each point along the streamlines:
        metric_vals = map_coordinates(metric_data, sls_pts.T, order=1, mode="constant")
        label_vals  = map_coordinates(labels_data, sls_pts.T, order=0, mode="constant").astype(int)

        ## Save the point-wise features to HDF5:
        df = pd.DataFrame({
            "subjectID" : args.subject,
            "bundle" : args.bundle,
            "segment" : label_vals,
            "value": metric_vals
        })
        df = df[df.segment != 0]
        print(df.segment.unique())

        if args.pointwise:
            hdf5_path = f"{args.output}/{args.subject}_{args.bundle}_streamlines_pointwise.h5"
            df.to_hdf(hdf5_path, key='df', mode='w', complevel=9)
            logging.info(f"Saved streamlines point-wise features to {hdf5_path}")
        else:
            logging.info("Skipping saving point-wise features as --pointwise not set to True")

        ## Compute mean for each segment:
        df_mean = (
            df.groupby(["subjectID", "bundle", "segment"], as_index=False)
            .agg(mean=("value", "mean"))
        )
        csv_path = f"{args.output}/{args.subject}_{args.bundle}_streamlines_average.csv"
        df_mean.to_csv(csv_path, index=False)
        logging.info(f"Saved streamlines segment-wise mean features to {csv_path}")

    except Exception as e:
        logging.error(f'An error occurred while processing {args.mask}: {str(e)}')

