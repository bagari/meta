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

#### Contact: Iyad Ba Gari <bagari@usc.edu>

If you use MeTA code, please cite the following publication:
* [Ba Gari, I., et al.: Heritability and Genetic Correlations Along the Corticospinal Tract. International Workshop on Computational Diffusion MRI. Cham: Springer Nature Morocco, 2024](https://doi.org/10.1007/978-3-031-86920-4_18)
* [Ba Gari, I., et al.: Medial tractography analysis (MeTA) for white matter population analyses across datasets. In: 2023 11th International IEEE/EMBS Conference on Neural Engineering (NER). pp. 1–5 (Apr 2023)](https://doi.org/10.1109/NER52421.2023.10123727)
* [Yushkevich, P.A.: Continuous medial representation of brain structures using the biharmonic PDE. Neuroimage 45(1 Suppl), S99–110 (Mar 2009)](https://doi.org/10.1016/j.neuroimage.2008.10.051)

## Installation

There are two options to use the package: via Conda or Docker/Singularity.

### Conda Installation

Create an environment with Python version >=3.9 and <3.12. For example:

```bash
conda config --add channels bioconda
conda create -n meta python==3.10
conda install bioconda::meta-neuro
```
> NOTE: Use `meta --help` to see the package options.


### Docker/Singularity Installation
To pull the singularity image:
```bash
singularity pull docker://quay.io/biocontainers/meta-neuro:1.0.1--py39hb5914e5_0
singularity pull docker://quay.io/biocontainers/meta-neuro:1.0.1--py310h6ccb7bc_0
singularity pull docker://quay.io/biocontainers/meta-neuro:1.0.1--py311h62e25fe_0
```

To execute the package with singularity:
```bash
singularity exec meta-neuro_1.0.1--py310h6ccb7bc_0.sif meta --help
```

## How to use the package:
* Convert streamlines in TRK format to a binary image:
```bash
meta_bundle_density --bundle CST.trk --reference dti_FA.nii.gz --output CST.nii.gz
```

* Generate a 3D Medial Surface for WM Bundle Using the CMREP Method: 
```bash
vtklevelset CST.nii.gz CST.vtk 0.1
cmrep_vskel -c 3 -p 1.5 -g CST.vtk CST_skeleton.vtk
```

* Run Medial Tractography Analysis (MeTA):
```bash
meta --subject 1234 --bundle CST --medial_surface CST_skeleton.vtk --volume CST.vtk --sbundle CST.trk --mbundle CST_model.trk --mask CST.nii.gz --num_segments 15 --output CST
```

* Extract Segment Features:
```bash
meta_segment_features --subject 1234 --bundle CST --mask CST_segments_local_core.nii.gz --map FA.nii.gz --output CST_FA_15_segments_local_core_metrics.csv
```

* Extract Streamline Features:
```bash
meta_streamlines_features --subject 1234 --bundle CST --mask CST.nii.gz --tractogram CST.trk --output CST_streamlines_metrics.csv
```
