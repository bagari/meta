import sys
import logging

import numpy as np
import vtk
from vtk.util import numpy_support
import pyvista as pv
import nibabel as nib
from scipy.spatial import Voronoi
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', encoding='utf-8', level=logging.INFO, force=True)


def _voxel_grid_image(data):
    """Wrap a 3D array as vtkImageData on the voxel grid."""
    nx, ny, nz = data.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    flat = np.asarray(data, dtype=np.float32).ravel(order='F')
    varr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    image.GetPointData().SetScalars(varr)
    return image


def _apply_affine(mesh, affine):
    """Transform mesh vertices by a 4x4 affine (voxel index -> world)."""

    mat = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            mat.SetElement(i, j, float(affine[i, j]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(mat)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(transform)
    tf.SetInputData(mesh)
    tf.Update()
    return tf.GetOutput()


def image_to_mesh(input_image, output_mesh=None, threshold=0.0,
                voxel_space=False, clean=False, preserve_labels=False,
                fix_normals=False, contour_2d=False):
    """Extract an iso-surface boundary mesh from a binary image."""

    img = nib.load(input_image)
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    image = _voxel_grid_image(data)

    logging.info(f"Image range: [{data.min()}, {data.max()}]; taking level set at {threshold}")

    if preserve_labels:
        logging.info("Preserving labels mode")
        append = vtk.vtkAppendPolyData()
        lbl = float(np.floor(threshold))
        while lbl <= data.max():
            dmc = vtk.vtkDiscreteMarchingCubes()
            dmc.SetInputData(image)
            dmc.ComputeGradientsOff()
            dmc.ComputeScalarsOff()
            dmc.ComputeNormalsOn()
            dmc.SetNumberOfContours(1)
            dmc.SetValue(0, lbl)
            dmc.Update()
            label_mesh = vtk.vtkPolyData()
            label_mesh.DeepCopy(dmc.GetOutput())
            scalar = numpy_support.numpy_to_vtk(
                np.full(label_mesh.GetNumberOfPoints(), lbl, dtype=np.uint16),
                deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
            scalar.SetName("Label")
            label_mesh.GetPointData().SetScalars(scalar)
            append.AddInputData(label_mesh)
            lbl += 1.0
        append.Update()
        surface = append.GetOutput()
    else:
        contour = vtk.vtkContourFilter() if contour_2d else vtk.vtkMarchingCubes()
        contour.SetInputData(image)
        contour.ComputeScalarsOff()
        contour.ComputeGradientsOff()
        contour.ComputeNormalsOn()
        contour.SetNumberOfContours(1)
        contour.SetValue(0, threshold)
        contour.Update()
        surface = contour.GetOutput()

    if clean:
        trifi = vtk.vtkTriangleFilter()
        trifi.SetInputData(surface)
        trifi.PassLinesOff()
        trifi.PassVertsOff()
        trifi.Update()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(trifi.GetOutputPort())
        cleaner.PointMergingOn()
        cleaner.SetTolerance(0.0)
        cleaner.Update()
        surface = cleaner.GetOutput()

    mesh = surface if voxel_space else _apply_affine(surface, affine)

    # A left-handed affine (det < 0) flips the surface, so flip the normals back.
    det = np.linalg.det(affine[:3, :3])
    if not voxel_space and (fix_normals or det < 0):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(mesh)
        if fix_normals:
            logging.info("vtkPolyDataNormals: auto-orienting normal vectors")
            normals.SetAutoOrientNormals(True)
        else:
            logging.info("vtkPolyDataNormals: inverting normal vectors due to sform")
            normals.SetFlipNormals(True)
        normals.Update()
        mesh = normals.GetOutput()

    mesh = pv.wrap(mesh)
    logging.info(f"Boundary mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    if output_mesh is not None:
        mesh.save(output_mesh)
        logging.info(f"Wrote boundary mesh to {output_mesh}")
    return mesh


def _edge_graph(points, faces_tri, weighted=True):
    """Sparse adjacency of the boundary mesh: Euclidean lengths, or unit weights."""

    edges = np.vstack([faces_tri[:, [0, 1]], faces_tri[:, [1, 2]], faces_tri[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    w = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1) if weighted \
        else np.ones(len(edges))
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    return sp.csr_matrix((np.concatenate([w, w]), (rows, cols)), shape=(len(points),) * 2)


def _enclosed_points(verts, surface, tolerance):
    """Mask of which points fall inside the closed boundary surface."""

    sel = vtk.vtkSelectEnclosedPoints()
    sel.SetInputData(pv.PolyData(verts))
    sel.SetSurfaceData(surface)
    sel.SetTolerance(tolerance)
    sel.Update()
    arr = sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    return numpy_support.vtk_to_numpy(arr).astype(bool)


def _log_thickness(poly):
    """Report area-weighted mean thickness over triangular faces (informational)."""

    try:
        rad = poly.GetCellData().GetArray("Radius")
        if rad is None:
            return
        wrapped = pv.wrap(poly)
        area = thickness = 0.0
        for i in range(wrapped.n_cells):
            pts = wrapped.get_cell(i).points
            if len(pts) == 3:
                a = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
                area += a
                thickness += rad.GetTuple1(i) * a
        if area > 0:
            logging.info(f"Surface area: {area}")
            logging.info(f"Mean thickness: {thickness / area}")
    except Exception as exc:
        logging.debug(f"thickness report skipped: {exc}")


def voronoi_skeleton(input_mesh, output_skeleton=None, prune=2.0, edge_degrees=0,
                    n_components=0, n_bins=0, sub_level=0, sub_mode="linear",
                    search_tol=1e-6):
    """Skeletonize a boundary mesh into a pruned medial surface."""

    bnd = (input_mesh if isinstance(input_mesh, pv.DataSet) else pv.read(input_mesh)).triangulate().clean(tolerance=1e-4)
    if sub_level > 0:
        bnd = bnd.subdivide(sub_level, subfilter="loop" if sub_mode == "loop" else "linear")

    points = np.asarray(bnd.points, dtype=float)
    faces_tri = bnd.faces.reshape(-1, 4)[:, 1:4]
    bounds = np.asarray(bnd.bounds, dtype=float)
    logging.info("Bounding Box : %f %f %f %f %f %f" % tuple(bounds))

    vor = Voronoi(points)
    vverts = vor.vertices
    nv = len(vverts)
    logging.info(f"Voronoi diagram: {nv} vertices, {len(vor.ridge_vertices)} faces")

    in_box = np.all((vverts >= bounds[0::2]) & (vverts <= bounds[1::2]), axis=1)
    if search_tol > 0 and nv > 0:
        logging.info(f"Selecting points inside mesh (n = {nv})")
        ptin = in_box & _enclosed_points(vverts, bnd, search_tol)
    else:
        ptin = in_box

    candidates = []
    for k, vids in enumerate(vor.ridge_vertices):
        if -1 in vids:
            continue
        vids = np.asarray(vids, dtype=np.int64)
        if ptin[vids].all():
            candidates.append((int(vor.ridge_points[k, 0]), int(vor.ridge_points[k, 1]), vids))

    sources = np.unique([c[0] for c in candidates]) if candidates else np.array([], dtype=np.int64)
    src_row = {int(s): i for i, s in enumerate(sources)}
    dist_geo = dist_edge = None
    if len(sources):
        dist_geo = dijkstra(_edge_graph(points, faces_tri), directed=False, indices=sources)
        if edge_degrees > 0:
            dist_edge = dijkstra(_edge_graph(points, faces_tri, weighted=False), directed=False, indices=sources)

    point_radius = np.full(nv, np.nan)
    cells, cell_radius, cell_geod, cell_prune = [], [], [], []
    npruned_edge = npruned_geo = 0

    for ip1, ip2, vids in candidates:
        row = src_row[ip1]
        if edge_degrees > 0 and dist_edge[row, ip2] < edge_degrees:
            npruned_edge += 1
            continue

        r = float(np.linalg.norm(points[ip1] - points[ip2]))
        dgeo = float(dist_geo[row, ip2])
        if dgeo < r * prune:
            npruned_geo += 1
            continue

        cells.append(vids)
        cell_radius.append(r)
        cell_geod.append(dgeo)
        cell_prune.append(dgeo / r if r > 0 else 0.0)

        # Inscribed-ball radius at each vertex:
        for v, rv in zip(vids, np.linalg.norm(vverts[vids] - points[ip1], axis=1)):
            if np.isnan(point_radius[v]):
                point_radius[v] = rv

    logging.info(f"Edge constraint pruned {npruned_edge} faces.")
    logging.info(f"Geodesic/Euclidean ratio constraint ({prune}) pruned {npruned_geo} faces.")

    skel = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(vverts), deep=True))
    skel.SetPoints(vpts)

    polys = vtk.vtkCellArray()
    for vids in cells:
        polys.InsertNextCell(len(vids))
        for v in vids:
            polys.InsertCellPoint(int(v))
    skel.SetPolys(polys)

    for values, name in [(cell_radius, "Radius"), (cell_geod, "Geodesic"), (cell_prune, "Pruning Ratio")]:
        arr = numpy_support.numpy_to_vtk(np.asarray(values, dtype=float), deep=True)
        arr.SetName(name)
        skel.GetCellData().AddArray(arr)
    radius_arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(point_radius), deep=True)
    radius_arr.SetName("VoronoiRadius")
    skel.GetPointData().AddArray(radius_arr)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(skel)
    cleaner.Update()
    poly = cleaner.GetOutput()
    logging.info(f"Clean filter: trimmed {nv} vertices to {poly.GetNumberOfPoints()}")

    if n_components > 0:
        connect = vtk.vtkPolyDataConnectivityFilter()
        connect.SetInputData(poly)
        if n_components == 1:
            connect.SetExtractionModeToLargestRegion()
        else:
            connect.SetExtractionModeToSpecifiedRegions()
            connect.InitializeSpecifiedRegionList()
            for region in range(n_components):
                connect.AddSpecifiedRegion(region)
        connect.ScalarConnectivityOff()
        connect.Update()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(connect.GetOutputPort())
        cleaner.Update()
        poly = cleaner.GetOutput()

    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(poly)
    c2p.PassCellDataOn()
    c2p.Update()
    skelfinal = c2p.GetPolyDataOutput()
    _log_thickness(skelfinal)

    if n_bins > 0:
        b = np.asarray(skelfinal.GetPoints().GetBounds(), dtype=float)
        lengths = b[1::2] - b[0::2]
        binsize = lengths.max() / n_bins
        cluster = vtk.vtkQuadricClustering()
        cluster.SetNumberOfDivisions(*[int(np.ceil(l / binsize)) for l in lengths])
        cluster.SetInputData(skelfinal)
        cluster.SetCopyCellData(1)
        cluster.Update()
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputConnection(cluster.GetOutputPort())
        c2p.PassCellDataOn()
        c2p.Update()
        skelfinal = c2p.GetPolyDataOutput()
        logging.info(f"QuadClustering -> {skelfinal.GetNumberOfPoints()} points, " f"{skelfinal.GetNumberOfCells()} cells")

    skelfinal = pv.wrap(skelfinal)
    if output_skeleton is not None:
        skelfinal.save(output_skeleton)
        logging.info(f"Wrote medial surface to {output_skeleton}")
    return skelfinal


def generate_medial_surface(input_image, output_mesh, output_skeleton,
                            threshold=0.0, prune=2.0, **skeleton_kwargs):
    """Image -> boundary mesh -> medial surface in one call; writes both files."""
    boundary = image_to_mesh(input_image, output_mesh, threshold=threshold)
    skeleton = voronoi_skeleton(boundary, output_skeleton, prune=prune, **skeleton_kwargs)
    return boundary, skeleton
