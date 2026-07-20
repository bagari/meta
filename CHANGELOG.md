# Changelog

### [2.1.0] - 2026-07-20

**Highlights**
- Added `apply_transform_image` CLI for DSI-Studio/ANTs image warping between subject and MNI spaces.
- Added `apply_transform_tractogram` CLI for affine and non-linear tractogram transforms.

**Tractogram and Image transforms**
- Added affine and non-linear tractogram transform support in `meta.transforms.tractogram`.
- Added support for `.mat`, `.txt`, `.npy`, and DSI-Studio `.mz` affine transforms.
- Added support for ANTs and DSI-Studio warp conventions through `--warp_source`.
- Added `--warp_first` to control transform order.
- Added `--no_trim` to keep streamline endpoints outside the warp grid.
- Added endpoint trimming for non-linear warp application, with reporting of trimmed points and dropped streamlines.
- Preserved TinyTrack conversion from DSI-Studio LPS voxel coordinates into RAS-mm space during reading.
- Added DSI-Studio image warping using `1InverseWarp.nii.gz` for subject-to-MNI and `1Warp.nii.gz` for MNI-to-subject.

---

### [2.0.1] - 2025-11-02
- Fixed bug in reorienting streamlines.
- Updated CM-Rep code to be compatible with latest update of VTK and ITK.

### [2.0.0] - 2025-10-02
- Added support for multiple white matter bundle formats: `trk`, `trx`, `tt.gz`, `tck`.  
- Added option to invert the transform matrix.  
- Fixed along-length segmentation/labeling issue when subject bundles were cropped or incomplete.  
- Added option to extract volumetric and streamline profiles.  
- Added streamline shape metrics.

---

### [1.0.1] - 2025-02-21
- Fixed issue where `excluded_idx` was empty, preventing along-length segmentation.  
- Prevented creation of empty DataFrames when computing bundle features from binary masks and microstructure maps.  

---

### [1.0.0] - 2024-11-02
Initial release of **Medial Tractography Analysis (MeTA)**: a workflow for minimizing brain microstructural heterogeneity in diffusion MRI (dMRI) metrics by extracting and parcellating the core volume along the length of white matter bundles in voxel space.  

**Features:**  
- Extraction of the medial surface and core volume of white matter (WM) bundles.  
- Segmentation of WM bundles along their length in voxel space.  
- Computation of volumetric and streamline-based features.  
