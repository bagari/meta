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



### Installation
MeTA supports Python version >=3.11 and <3.14.
Using **pip**:
```sh
pip install meta-neuro
```

Using **Bioconda**:
```sh
conda config --add channels bioconda
conda install bioconda::meta-neuro
```

## Usage

```sh
## Using DSI Studio transforms:
# Compute density map and convert streamlines to a binary image:
density_map --tractogram subject_CST.tt.gz --reference subject_FA.nii.gz \
            --output "output_dir/subjectID_CST.nii.gz"

# Medial Tractography Analysis (MeTA):
meta --subject "subjectID_12345" --bundle "CST" \
    --mbundle "model_CST.tt.gz" --sbundle "subject_CST.tt.gz" \
    --mask "output_dir/subjectID_CST.nii.gz" \
    --warp "subjectID.1InverseWarp.nii.gz" --warp_source "dsi_studio" \
    --seg_method "hyperplane" --num_segments 15 --output "output_dir"

# Extract Voxel-based Bundle Profile: Compute volumetric profile for DTI maps e.g., FA, MD, RD, AD, etc. Output two files: 1) *_segments_average.csv file with the average profile along the bundle length, and 2) *_segments_voxelwise.h5 (with option `--voxelwise `): the profile for each voxel in the bundle.
volumetric_profile --subject "subjectID_12345" --bundle "CST" --mask CST_local_all.nii.gz --map subject_FA.nii.gz --output "output_dir"

# Extract Streamline-based Profile: Compute streamline profile based on tractography and DTI maps e.g., FA, MD, RD, AD, etc. output two files: 1) *_streamlines_average.csv file with the average profile along the bundle length, and 2) *_streamlines_pointwise.h5 (with option `--pointwise `): the profile for each point of streamline.

streamlines_profile --subject "subjectID_12345" --bundle "CST" --tractogram "subject_CST.tt.gz" --mask CST_local_all.nii.gz --map subject_FA.nii.gz --output "output_dir"

# Extract Bundle Shape Features: Bundle shape features implemented based on Yeh et al., 2020. The following features are extracted: Total number of streamlines, Average streamlines length, Span, Curl, Volume, Surface area, Diameter, Elongation, Irregularity


shape_metrics --subject "subjectID_12345" --bundle "CST" --mask CST_local_all.nii.gz --tractogram "subject_CST.tt.gz" --output CST_streamlines_metrics.csv
```

<!-- ```sh
## Using ANTs transforms:

``` -->
