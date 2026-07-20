import sys
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from meta.io.streamline import read_streamlines
from dipy.tracking.streamline import length

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)

def bundle_length(streamlines):
    return np.mean([length(sl) for sl in streamlines])

def get_span(streamlines):
    spans = [np.linalg.norm(line[0]-line[-1]) for line in streamlines]
    return np.mean(spans)
def get_curl(streamlines):
    return (bundle_length(streamlines) / get_span(streamlines))

def get_volume(img):
    data = img.get_fdata()
    voxel_dimensions = img.header.get_zooms()
    voxel_volume = np.prod(voxel_dimensions[:3]) or 1
    return (data > 0).sum() * voxel_volume

def get_surface_area(img):
    data = img.get_fdata()
    voxel_dimensions = img.header.get_zooms()
    indices = np.where(data > 0)
    neighborhood = list(itertools.product([-1, 0, 1], repeat=3))
    padded_data = np.pad(data, pad_width=1)
    surface_voxels = sum(
        any(padded_data[x + dx + 1, y + dy + 1, z + dz + 1] == 0 for dx, dy, dz in neighborhood)
        for x, y, z in zip(*indices))  
    voxel_area = np.prod(voxel_dimensions[:2]) or 1
    return surface_voxels * voxel_area

def get_diameter(streamlines, img):
    return 2*np.sqrt(get_volume(img)/(np.pi*bundle_length(streamlines)))

def get_elongation(streamlines, img):
    return bundle_length(streamlines) / get_diameter(streamlines, img)

def get_irregularity(streamlines, img):
    return get_surface_area(img) / (np.pi * get_diameter(streamlines, img) * bundle_length(streamlines))


## Compute streamlines shape metrics:
def shape_features():
    parser = argparse.ArgumentParser(description='Extract shape features from white matter bundle streamlines')
    parser.add_argument('--subject', type=str, help='Subject ID', required=True)
    parser.add_argument('--bundle', type=str, help='White matter bundle name', required=True)
    parser.add_argument('--mask', type=str, help='Binary image of white matter bundle', required=True)
    parser.add_argument('--tractogram', type=str, help='Streamline file of white matter bundle', required=True)
    parser.add_argument('--output', type=str, help='Output directory to save features', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    mask = nib.load(args.mask)

    streamlines, _, _, _ = read_streamlines(args.tractogram)
    
    streamlines_count = len(streamlines)
    streamlines_length = bundle_length(streamlines)
    span = get_span(streamlines)
    curl = get_curl(streamlines)
    volume = get_volume(mask)
    surface_area = get_surface_area(mask)
    diameter = get_diameter(streamlines, mask)
    elongation = get_elongation(streamlines, mask)
    irregularity = get_irregularity(streamlines, mask)

    data = {
        'subjectID': args.subject,
        'bundle': args.bundle,
        'streamlines_count': streamlines_count,
        'length': streamlines_length,
        'span': span,
        'curl': curl,
        'volume': volume,
        'surface_area': surface_area,
        'diameter': diameter,
        'elongation': elongation,
        'irregularity': irregularity
    }

    features = pd.DataFrame(data, index=[0])

    if args.output.endswith('.csv'):
        features.to_csv(args.output, index=False)
    else:
        features.to_csv(f"{args.output}/{args.subject}_{args.bundle}_shape_metrics.csv", index=False)

    logging.info('Finished extracting shape features')


