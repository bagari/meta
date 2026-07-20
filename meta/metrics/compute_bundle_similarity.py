import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr
from meta.transforms.image import warp_image_dsi_studio

## https://tractometer.org/tractometer/the_metrics/
def precision(mask_pred, mask_reference):
    """Return the fraction of predicted voxels that overlap the reference mask."""
    tp = np.sum(np.logical_and(mask_pred, mask_reference))
    den = np.sum(mask_pred)
    return np.nan if den == 0 else tp / den

def recall(mask_pred, mask_reference):
    """Return bundle overlap: fraction of reference voxels recovered by prediction."""
    tp = np.sum(np.logical_and(mask_pred, mask_reference))
    den = np.sum(mask_reference)
    return np.nan if den == 0 else tp / den

def dice(mask_pred, mask_reference):
    """Return the Dice similarity coefficient between predicted and reference masks."""
    tp = np.sum(np.logical_and(mask_pred, mask_reference))
    den = np.sum(mask_pred) + np.sum(mask_reference)
    return np.nan if den == 0 else 2 * tp / den

def overreach_vs(mask_pred, mask_reference):
    """Return false-positive voxels normalized by predicted bundle volume."""
    fp = np.sum(np.logical_and(mask_pred, np.logical_not(mask_reference)))
    den = np.sum(mask_pred)
    return np.nan if den == 0 else fp / den

def overreach_reference(mask_pred, mask_reference):
    """Return false-positive voxels normalized by reference bundle volume."""
    fp = np.sum(np.logical_and(mask_pred, np.logical_not(mask_reference)))
    den = np.sum(mask_reference)
    return np.nan if den == 0 else fp / den

def jaccard(mask_pred, mask_reference):
    """Return the Jaccard index between predicted and reference masks."""
    intersection = np.logical_and(mask_pred, mask_reference)
    union = np.logical_or(mask_pred, mask_reference)

    den = np.sum(union)
    return np.nan if den == 0 else np.sum(intersection) / den


## Mean Absolute Error (MAE)
def mae(map_1, map_2):
    """Return the mean absolute error between two density maps."""
    return np.mean(np.abs(map_1 - map_2))

## Mean Squared Error (MSE)
def mse(map_1, map_2):
    """Return the mean squared error between two density maps."""
    return np.mean((map_1 - map_2) ** 2)

## Root Mean Squared Error (RMSE)
def rmse(map_1, map_2):
    """Return the root mean squared error between two density maps."""
    return np.sqrt(mse(map_1, map_2))

## Pearson Correlation Coefficient (PCC) or Normalized Cross Correlation coefficient (NCC)
EPS = 1e-12
def ncc(map_1, map_2):
    """Return normalized cross-correlation, equivalent to Pearson correlation."""
    x = map_1.flatten()
    y = map_2.flatten()
    if len(x) < 2:
        return np.nan
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)
    if x_std < EPS or y_std < EPS:
        return np.nan
    x = (x - np.mean(x)) / x_std
    y = (y - np.mean(y)) / y_std
    return np.sum(x * y) / (len(x) - 1)

## Spearman rank correlation
def spearman(map_1, map_2):
    """Return Spearman rank correlation between two density maps."""
    x = map_1.flatten()
    y = map_2.flatten()
    if len(x) < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return np.nan
    rho, _ = spearmanr(x, y)
    return rho

def nmi(map_1, map_2, bins=64):
    """Return normalized mutual information estimated from a joint histogram."""
    map_1 = map_1.flatten()
    map_2 = map_2.flatten()
    joint_hist, _, _ = np.histogram2d(map_1, map_2, bins=bins)
    hist_sum = np.sum(joint_hist)
    if hist_sum == 0:
        return np.nan
    pxy = joint_hist / hist_sum
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    h_x = -np.sum(px[px > 0] * np.log2(px[px > 0]))
    h_y = -np.sum(py[py > 0] * np.log2(py[py > 0]))
    h_xy = -np.sum(pxy[pxy > 0] * np.log2(pxy[pxy > 0]))
    return np.nan if h_xy < EPS else (h_x + h_y) / h_xy


def main():
    parser = argparse.ArgumentParser(description="Compute similarity metrics between an input bundle image and a reference atlas image.")
    parser.add_argument("--subject", type=str, help="Subject ID.")
    parser.add_argument("--bundle", type=str, help="Name of white matter bundle.")
    parser.add_argument("--input", required=True, help="Input NIfTI image.")
    parser.add_argument("--references", nargs="+", required=True, help="Reference NIfTI image in MNI space.")
    parser.add_argument("--output-csv", required=True, help="Output CSV.")

    parser.add_argument("--warp", help="DSI-Studio warp to move input into reference space.")
    parser.add_argument("--warp_ref", help="Reference NIfTI whose grid/affine defines the output space.")
    parser.add_argument("--order", type=int, default=1, choices=[0, 1, 3, 5])
    parser.add_argument("--cval", type=float, default=0.0)
    parser.add_argument("--save-warped", dest="save_warped", action="store_true", help="Save warped input image next to the output CSV.")

    parser.add_argument("--percentile",type=int, default=0, help="Threshold the reference by a percentile value (1-100).")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--nmi-bins", type=int, default=64)
    args = parser.parse_args()

    input_image = nib.load(args.input)

    ## using multiple references
    for reference_path in args.references:
        reference_image = nib.load(reference_path)
        same_grid = (input_image.shape == reference_image.shape and np.allclose(input_image.affine, reference_image.affine, atol=1e-5))
        if same_grid:
            print(f"Using refence:{reference_path}")
            break

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    warped_input_path = None
    if same_grid:
        metric_image = input_image
    else:
        if args.warp is None:
            raise ValueError("Input and reference are not in the same space. Pass --warp to move the input image into reference space.")

        if args.save_warped:
            output_name = os.path.splitext(os.path.basename(args.output_csv))[0]
            warped_input_path = os.path.join(output_dir, output_name + "_input_in_reference_space.nii.gz")

        metric_image = warp_image_dsi_studio(image=args.input, warp=args.warp, warp_ref=args.warp_ref, output=warped_input_path, order=args.order, fill_value=args.cval)

    input_density = metric_image.get_fdata()
    reference_density = reference_image.get_fdata()
    threshold_ref = np.percentile(reference_density.flatten()[reference_density.flatten()>0], args.percentile)
    print(f"Thresholding reference by {threshold_ref} ...")
    reference_density = reference_density > np.percentile(threshold_ref, args.percentile)

    input_mask = input_density > args.threshold
    reference_mask = reference_density > args.threshold

    row = {
        "subjectID": args.subject,
        "bundle": args.bundle,
        "dice": dice(input_mask, reference_mask),
        "overlap": recall(input_mask, reference_mask),
        "precision": precision(input_mask, reference_mask),
        "overreach_reference": overreach_reference(input_mask, reference_mask),
        "overreach_predicted": overreach_vs(input_mask, reference_mask),
        "jaccard": jaccard(input_mask, reference_mask),
        "ncc": ncc(input_density, reference_density),
        "spearman": spearman(input_density, reference_density),
        "nmi": nmi(input_density, reference_density, bins=args.nmi_bins),
        "mae": mae(input_density, reference_density),
        "mse": mse(input_density, reference_density),
        "rmse": rmse(input_density, reference_density),
    }

    pd.DataFrame([row]).to_csv(args.output_csv, index=False)

    print(f"Done: {args.output_csv}")


if __name__ == "__main__":
    main()
