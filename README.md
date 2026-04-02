![GitHub release (latest by date)](https://img.shields.io/github/v/release/bagari/meta?logo=Github)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/meta-neuro/README.html)
![Platforms](https://anaconda.org/bioconda/meta-neuro/badges/platforms.svg)
![Downloads](https://img.shields.io/conda/dn/bioconda/meta-neuro)
![License](https://anaconda.org/bioconda/meta-neuro/badges/license.svg)


## Medial Tractography Analysis (MeTA)

<p align="center">
<img width="800" alt="workflow" src="https://github.com/bagari/meta/blob/main/resources/MeTA_workflow.png">
</p>

MeTA is a workflow implemented to minimize microstructural heterogeneity in diffusion MRI (dMRI) metrics by extracting and parcellating the core volume along the bundle length in the voxel-space directly while effectively preserving bundle shape and efficiently capturing the regional variation within and along white matter (WM) bundles.

#### Contact: Iyad Ba Gari <iyad.bagari@usc.edu>

If you use MeTA code, please cite the following publication:
* [Ba Gari, I., et al.: Heritability and Genetic Correlations Along the Corticospinal Tract. International Workshop on Computational Diffusion MRI. Cham: Springer Nature Morocco, 2024](https://doi.org/10.1007/978-3-031-86920-4_18)
* [Ba Gari, I., et al.: Along-tract parameterization of white matter microstructure using medial tractography analysis (MeTA). In: The 19th International Symposium on Medical Information Processing and Analysis (2023)](https://doi.org/10.1109/SIPAIM56729.2023.10373540)

## Installation

There are two options to use the package: via Conda or Docker/Singularity.

### Conda Installation

Create an environment with Python version >=3.9 and <3.12. For example:

```bash
conda config --add channels bioconda
conda create -n meta python==3.13
conda activate meta
conda install bioconda::meta-neuro=2.0.1
```

### Singularity Installation
To pull the singularity image using apptainer:
```bash
apptainer pull meta_2_0_1.sif docker://quay.io/biocontainers/meta-neuro:2.0.1--py313h47f2c4e_0

apptainer run meta_2_0_1.sif meta --help
```
> NOTE: Use `meta --help` to see the package options.


## How to use the package:

### Generate Medial Surface for WM Bundle:
Medial surface is extract based on Continuous medial representation (CMREP) method [Yushkevich, 2009](https://doi.org/10.1016/j.neuroimage.2008.10.051).
* Convert streamlines in trk/tck/tt.gz formats to a binary image.

```bash
density_map --tractogram CST.trk --reference dti_FA.nii.gz --output CST.nii.gz
```

* Generate a 3D Medial Surface for WM Bundle using the CMREP Method:

```bash
vtklevelset CST.nii.gz CST.vtk 0.1
cmrep_vskel -c 3 -p 1.5 -g CST.vtk CST_skeleton.vtk
```


### Run Medial Tractography Analysis (MeTA):
MeTA will extract the core volume of WM bundle and parcellate it into segments along the bundle length.
```bash
meta --subject 1234 --bundle CST --medial_surface CST_skeleton.vtk --volume CST.vtk --sbundle CST.trk --mbundle CST_model.trk --transform subject_ANTs0GenericAffine.mat --mask CST.nii.gz --num_segments 15 --output CST
```


### Extract Voxel-based Bundle Profile:
Compute volumetric profile based on binary masks and microstructure maps e.g., FA, MD, RD, AD, etc. Output two files: 1) *_segments_average.csv file with the average profile along the bundle length, and 2) *_segments_voxelwise.h5: the profile for each voxel in the bundle.

```bash
volumetric_profile --subject 1234 --bundle CST --mask CST_local_all.nii.gz --map FA.nii.gz --output /output_folder
```

### Extract Streamline-based Profile:
Compute streamline profile based on tractography and microstructure maps e.g., FA, MD, RD, AD, etc. output two files: 1) *_streamlines_average.csv file with the average profile along the bundle length, and 2) *_streamlines_pointwise.h5: the profile for each point of streamline.

```bash
streamlines_profile --subject 1234 --bundle CST --tractogram CST.trk --mask CST_local_all.nii.gz --map FA.nii.gz --output /output_folder
```


### Extract Bundle Shape Features:
Bundle shape features implemented based on [Yeh et al., 2020](https://doi.org/10.1016/j.neuroimage.2020.117329). The following features are extracted:
1. Total number of streamlines
2. Average streamlines length
3. Span
4. Curl
5. Volume
6. Surface area
7. Diameter
8. Elongation
9. Irregularity

```bash
shape_metrics --subject 1234 --bundle CST --mask CST.nii.gz --tractogram CST.trk --output CST_streamlines_metrics.csv
```
