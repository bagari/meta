import os
import io
import gzip
import numpy as np
from scipy.io import loadmat


# Adapted from: https://github.com/scilus/scilpy/blob/48911befe7049711536c7dd649e385ca30a9bbc0/scilpy/io/utils.py#L802
# Added DSI-Studio .mz support
def load_transformation(affine_path):
    """
    Load a transformation matrix from a file.

    Parameters:
        affine_path: Path to the transformation file.

    Returns:
        data: 4x4 transformation matrix.
    """
    _, ext = os.path.splitext(affine_path)
    ext = ext.lower()

    if ext == '.txt':
        data = np.loadtxt(affine_path)

    elif ext == '.npy':
        data = np.load(affine_path)

    elif ext == '.mat':
        transfo_dict = loadmat(affine_path)
        lps2ras = np.diag([-1, -1, 1])
        transfo_key = 'AffineTransform_double_3_3'
        if transfo_key not in transfo_dict:
            transfo_key = 'AffineTransform_float_3_3'
        rot = transfo_dict[transfo_key][0:9].reshape((3, 3))
        trans = transfo_dict[transfo_key][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

        data = np.eye(4)
        data[0:3, 3] = r_trans
        data[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)

    elif ext == '.mz':
        with gzip.open(affine_path, 'rb') as f:
            data = f.read()
        mat = loadmat(io.BytesIO(data), appendmat=False)
        if 'T' not in mat:
            raise ValueError("DSI-Studio .mz transform does not contain key 'T'. Please update DSI-Studio.")
        data = mat['T'].T

    else:
        raise ValueError('Extension {} is not supported'.format(ext))
    return data
