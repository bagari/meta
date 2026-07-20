import io
import sys
import gzip
import logging
import numpy as np
import scipy.io as sio

import nibabel as nib
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.tck import TckFile

import trx.trx_file_memmap as tmm
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import transform_streamlines, Streamlines

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)

def parse_tt(tinytrack):
    """
    Parse DSI-Studio TinyTrack format (.tt.gz) to extract streamlines and metadata.

    Parameters:
        tinytrack: Path to the TinyTrack file.

    Returns:
        streamlines: List of streamlines in LPS space.
        tt_affine: Affine transformation matrix.
        dimension: Dimensions of image.
        voxel_size: Voxel size of image.
        voxel_order: Voxel order of image.
    """

    if tinytrack.endswith('.tt.gz'):
        with gzip.open(tinytrack, 'rb') as f:
            data = f.read()
        mat = sio.loadmat(io.BytesIO(data), appendmat=False)
    tt_affine = mat['trans_to_mni'].reshape(4,4)
    dimension = tuple(mat['dimension'].ravel().astype(int))
    voxel_size = tuple(mat['voxel_size'].ravel().astype(float))
    voxel_order = "".join(nib.aff2axcodes(tt_affine))

    buf1 = mat.get('track').flatten()
    buf2 = buf1.view(np.int8)
    length = len(buf1)
    pos = []
    i = 0
    while i < length:
        pos.append(i)
        track_length = np.frombuffer(buf1[i:i+4].tobytes(), dtype=np.uint32)[0]
        i += int(track_length) + 13

    streamlines = []
    for p in pos:
        size_val = np.frombuffer(buf1[p:p+4].tobytes(), dtype=np.uint32)[0]
        num_points = int(size_val // 3)
        x = np.frombuffer(buf1[p+4:p+8].tobytes(), dtype=np.int32)[0]
        y = np.frombuffer(buf1[p+8:p+12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[p+12:p+16].tobytes(), dtype=np.int32)[0]

        track_pts = np.empty((num_points, 3), dtype=np.float32)
        track_pts[0, :] = [x, y, z]
        p_offset = p + 16

        for j in range(1, num_points):
            dx = int(buf2[p_offset])
            dy = int(buf2[p_offset + 1])
            dz = int(buf2[p_offset + 2])
            x += dx; y += dy; z += dz
            track_pts[j, :] = [x, y, z]
            p_offset += 3
        track_pts /= 32.0
        streamlines.append(track_pts)

    logging.info(f"Parsed {len(streamlines)} streamlines in total")
    logging.info(f"Affine: \n {tt_affine},\n Dimension: {dimension},\n Voxel Size: {voxel_size},\n Voxel Order: {voxel_order}")
    return streamlines, tt_affine, dimension, voxel_size, voxel_order


def read_streamlines(bundle_path, reference=None):
    """
    Read streamlines from TRK, TCK, TRX, or TinyTrack format.
    
    Parameters:
        bundle_path: Path to segmented white matter bundle file.

    Returns:
        streamlines: List of streamlines in RASMM space.
    """
    ## TCK format:
    if bundle_path.endswith('.tck'):
        bundle = load_tractogram(bundle_path, reference, bbox_valid_check=False)
        streamlines = bundle.streamlines
        groups = getattr(bundle, "groups", {})
        affine = getattr(bundle, "affine", np.eye(4))
        dimension = getattr(bundle, "dimensions", None)

        return streamlines, groups, affine, dimension

    ## TRK format:
    if bundle_path.endswith(('.trk')):
        bundle =  load_tractogram(bundle_path, "same", bbox_valid_check=False)
        streamlines = bundle.streamlines
        groups = getattr(bundle, "groups", {})
        affine = getattr(bundle, "affine", np.eye(4))
        dimension = getattr(bundle, "dimensions", None)

        return streamlines, groups, affine, dimension

    ## TRX format:
    if bundle_path.endswith(('.trx')):
        print("Loading .trx bundle...") 
        bundle = tmm.load(bundle_path)
        streamlines = bundle.streamlines
        groups = getattr(bundle, "groups", {})
        affine = getattr(bundle, "affine", np.eye(4))
        dimension = getattr(bundle, "dimensions", None)

        return streamlines, groups, affine, dimension

    if bundle_path.endswith('.tt.gz'):
        ## Read DSI-Studio format (TinyTrack):
        streamlines, tt_affine, dimension, _, _ = parse_tt(bundle_path)
        streamlines_lps_center = [s - 0.5 for s in streamlines]
        streamlines = transform_streamlines(streamlines_lps_center, tt_affine)
        streamlines = Streamlines(streamlines)
        groups = {}
        affine = tt_affine
        dimension = tuple(dimension) if dimension is not None else None

        return streamlines, groups, affine, dimension
    
    raise ValueError(f"Supported formats: .trk, .tck, .trx, .tt.gz")


def convert_tinytrack(bundle_path, output_format='trk'):
    """
    Convert TinyTrack (.tt.gz) to TRK or TCK format.

    Args:
        bundle_path: Path to segmented white matter bundle file in TinyTrack format.
        output_format: Output format, either 'trk' or 'tck'.
    """

    if bundle_path.endswith('.tt.gz'):
        output = bundle_path[:-6]
    else:
        output = bundle_path

    ## Read TinyTrack:
    streamlines, tt_affine, dimension, voxel_size, voxel_order = parse_tt(bundle_path)
    streamlines_lps_center = [s - 0.5 for s in streamlines]
    tractogram = Tractogram(streamlines_lps_center, affine_to_rasmm=tt_affine)

    if output_format == 'tck':
        tck = TckFile(tractogram=tractogram)
        tck.save(f'{output}.tck')
    else:
        header = {}
        header["dimensions"] = dimension
        header["voxel_sizes"] = voxel_size
        header["voxel_to_rasmm"] = tt_affine
        header["voxel_order"] = voxel_order
        header["magic_number"] = b"TRACK"
        header["nb_streamlines"] = len(streamlines_lps_center)
        header["version"] = 2
        header["hdr_size"] = 1000
        trk = TrkFile(tractogram, header=header)
        trk.save(f'{output}.trk') 

