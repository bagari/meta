import sys
import logging
import numpy as np
import pyvista as pv
import nibabel as nib
from scipy import ndimage
from nibabel.affines import apply_affine

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)


def process_mesh(mesh, fill_holes=True, fill_size=2500, get_largest=True):
    """
    Cleans and processes a mesh by filling small holes and extracting the largest connected component.

    Parameters:
        mesh: Input mesh object to process.
        fill_holes: If True, fill holes in the mesh up to the specified size.
        fill_size: Maximum size (in area) of holes to be filled.
        get_largest: If True, extract only the largest connected component of the mesh.
    """

    mesh.clean(inplace=True)
    if fill_holes:
        mesh.fill_holes(fill_size, inplace=True)
    if get_largest:
        mesh.extract_largest(inplace=True)
    return mesh.clean(lines_to_points=False, polys_to_lines=False, strips_to_polys=False)


def medial_core(bundle_mask, medial_surface, volume_mesh, percent=0.125, fill=True, size=2500, extract=False):
    """
    Computes the medial core of a white matter bundle based on the medial surface and volume mesh.

    Parameters:
        bundle_mask: Path to the bundle mask in NIfTI format.
        medial_surface: Path to themedial surface mesh in VTK format.
        volume_mesh: Path to the volume mesh in VTK format.
        percent: Percentage of distance to keep from each side of the medial surface.
        fill: If True, fills holes in the bundle mesh.
        size: Maximum size of holes to fill.
        extract: If True, extracts the largest connected component of the bundle mesh.

    Returns:
        core: NIfTI image of the medial core.
        average_depth: Average depth of the medial surface.
    """
    # Read medial surface and volume mesh
    surface = pv.PolyData(medial_surface)
    surface = process_mesh(surface, fill_holes=fill, fill_size=size, get_largest=extract)
    volume = pv.PolyData(volume_mesh)
    volume = process_mesh(volume, fill_holes=fill, fill_size=size, get_largest=extract)

    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError("Empty medial surface")
    
    # Load bundle mask:
    bundle = nib.load(bundle_mask)
    affine = bundle.affine
    inv_affine = np.linalg.inv(affine)
    mask_data = bundle.get_fdata()
    core_mask = np.zeros(bundle.shape, dtype=np.uint8)

    # Compute normals and distances on the medial surface:
    surf_normals = surface.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    points = surf_normals.points
    normals = surf_normals["Normals"]
    ray_vecs = normals * surf_normals.length

    n_points = surf_normals.n_points
    distances = np.full(n_points, np.nan, dtype=float)
    pct_pt_chunks = []

    for i in range(n_points):
        point = points[i]
        point_1 = point - ray_vecs[i]
        point_2 = point + ray_vecs[i]

        inter_pts, _ = volume.ray_trace(point_1, point_2, first_point=False)
        n_inter = inter_pts.shape[0]

        if n_inter == 0:
            continue

        diff = inter_pts - point
        dists = np.linalg.norm(diff, axis=1)
        distances[i] = dists.min()

        if n_inter >= 2:
            nearest_idx = np.argpartition(dists, 1)[:2]
            nearest_pts = inter_pts[nearest_idx]
            pct_pt_chunks.append(percent * nearest_pts + (1.0 - percent) * point)

    surf_normals["distances"] = distances
    average_depth = np.nanmean(distances)
    logging.info(f"Average depth: {average_depth}")

    if pct_pt_chunks:
        pct_pts = np.vstack(pct_pt_chunks)
        core_pts = np.vstack((pct_pts, surface.points))
    else:
        core_pts = surface.points.copy()

    vox_idx = np.rint(apply_affine(inv_affine, core_pts)).astype(np.int64)
    valid = np.all((vox_idx >= 0) & (vox_idx < np.array(core_mask.shape)), axis=1)
    vox_idx = vox_idx[valid]

    if vox_idx.size:
        core_mask[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = 1

    vox_size = np.mean(bundle.header.get_zooms()[:3])
    if vox_size < 1:
        core_pad = np.pad(core_mask, pad_width=3, mode="constant", constant_values=0)
        closed_pad = ndimage.binary_closing(core_pad, structure=np.ones((5, 5, 5)))
        core_closed = closed_pad[3:-3, 3:-3, 3:-3]
    else:
        core_pad = np.pad(core_mask, pad_width=1, mode="constant", constant_values=0)
        closed_pad = ndimage.binary_closing(core_pad, structure=np.ones((2, 2, 2)))
        core_closed = closed_pad[1:-1, 1:-1, 1:-1]

    labels, num_labels = ndimage.label(core_closed.astype(np.uint8))
    sizes = ndimage.sum(core_closed, labels, range(num_labels + 1))
    largest_label = np.argmax(sizes[1:]) + 1
    largest_component_mask = labels == largest_label
    largest_component_mask *= mask_data > 0

    core = nib.Nifti1Image(largest_component_mask.astype(np.uint8), affine)

    return core, average_depth
