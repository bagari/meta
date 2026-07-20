import sys
import logging
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)

def warp_image_dsi_studio(image, warp, warp_ref, output=None, order=1, fill_value=0.0):
    """
    Resample image onto reference with a DSI-Studio warp.
    Use 1InverseWarp for subject-to-MNI and 1Warp for MNI-to-subject.
    """
    def reorient(data, from_affine, to_affine):
        """Match data orientation to another affine."""
        from_ornt = io_orientation(from_affine)
        to_ornt = io_orientation(to_affine)
        if np.array_equal(from_ornt, to_ornt):
            return data
        xform = ornt_transform(from_ornt, to_ornt)
        return apply_orientation(data, xform)

    img = nib.load(image)
    image_dtype = img.get_data_dtype()
    data = np.asanyarray(img.dataobj)
    if order > 0:
        data = data.astype(np.float32, copy=False)

    warp_img = nib.load(warp)
    warp_data = warp_img.get_fdata(dtype=np.float32)
    if warp_data.ndim == 5:
        if warp_data.shape[3] != 1:
            raise ValueError(f"Expected 5D warp shape (X, Y, Z, 1, 3), got {warp_data.shape}.")
        warp_data = warp_data[:, :, :, 0, :]
    if warp_data.ndim != 4 or warp_data.shape[-1] != 3:
        raise ValueError(f"Expected warp shape (X, Y, Z, 3), got {warp_data.shape}.")

    ref_img = nib.load(warp_ref)
    if warp_img.shape[:3] != ref_img.shape[:3]:
        raise ValueError(f"Warp grid {warp_img.shape[:3]} must match reference grid "
            f"{ref_img.shape[:3]}. Use 1InverseWarp for subject-to-MNI or 1Warp for MNI-to-subject.")

    data = reorient(data, img.affine, warp_img.affine)

    coords = np.stack([warp_data[..., 0], warp_data[..., 1], warp_data[..., 2]], axis=0).astype(np.float32)
    warped = map_coordinates(data, coords, order=order, mode="constant", cval=fill_value, prefilter=(order > 1))
    warped = reorient(warped, warp_img.affine, ref_img.affine)

    dtype = image_dtype if order == 0 else np.float32
    out_data = np.asarray(warped, dtype=dtype)

    new_header = ref_img.header.copy()
    new_header.set_data_dtype(dtype)
    warped_img = nib.Nifti1Image(out_data, ref_img.affine, new_header)
    if output is not None:
        nib.save(warped_img, output)

    return warped_img

def apply_transform_image():
    parser = argparse.ArgumentParser(description="Resample a NIfTI image using a DSI-Studio non-linear warp.")
    parser.add_argument("--image", required=True, help="Input NIfTI image to resample.")
    parser.add_argument("--warp", required=True, help="DSI-Studio warp: 1InverseWarp.nii.gz (subject→MNI) or 1Warp.nii.gz (MNI→subject).")
    parser.add_argument("--warp_ref", required=True, help="Reference NIfTI whose grid/affine defines the output space.")
    parser.add_argument("--output", required=True, help="Output NIfTI filename.")
    parser.add_argument("--order", type=int, default=1, choices=[0, 1, 3, 5], help="Interpolation order: 0=nearest (labels), 1=trilinear (default, continuous), 3/5=cubic/quintic spline.")
    parser.add_argument("--cval", type=float, default=0.0, help="Fill value outside the source image.")
    args = parser.parse_args()

    warp_image_dsi_studio(image=args.image, warp=args.warp, warp_ref=args.warp_ref, output=args.output, order=args.order, fill_value=args.cval)
    logging.info(f"Resampled image saved to {args.output}")


if __name__ == "__main__":
    apply_transform_image()
