# Spatial Analysis Framework
A package for integrated spatial analysis for single-cell digital pathology.

**Unpublished and unreleased codebase, DO NOT DISTRIBUTE.**

@ Tina Hsu, 2025.

## Overview of the analysis framework
<img width="1074" height="617" alt="spatial-framework-fig" src="https://github.com/user-attachments/assets/25a26400-584a-4505-9beb-6df2b920e7ed" />

## Installation
Clone this repo
```commandline
git clone https://github.com/tina0420/Spatial-Analysis-Framework.git
```

## Dependencies
Here is a list of required libraries, and it is also listed in `requirements.txt`.

`numpy pandas scipy seaborn matplotlib scikit-learn anndata scanpy louvain squidpy shapely pathlib setuptools tifffile scikit-image xml readimc xtiff cv2 deepcell`

## Functions
All current functions are listed in this [page](FUNCTION_LIST.md).

## Quick Documentation
### Supporting inputs
1. **tiff** files: multiplexed image stacks.

2. **h5ad** files: segmented images in AnnData objects.

Modality specific file types of the following are supported:
- **mcd** files: machine files from Imaging Mass Cytometry Hyperion.

Examples of the above files are shown in `example/data/`.

### Example 1: Working with IMC data
 Below we outline an example workflow to analyze IMC data.
```commandline
python -m example.1a_processing_mcd_or_tiffs
```
This file first processes and segments the MCD files into processed AnnData objects using functions from `src.segment` and `src.preprocess`.

```commandline
python -m example.1b_clustering
```
This file denovo clusters cells by expression using functions from `src.annotate`.

```commandline
python -m example.1c_annotate
```
This file annotates the resulting clusters into meaningful cell types of cell states.

```commandline
python -m example.1d_visualization
```
This file exhibits a series of visualizations using functions from `src.visualize` to help with spatial analysis.

```commandline
python -m example.1e_cluster_evaluation
```
This file performs an optional analysis of comparing clustering methods using functions from `src.annotate_evaluation`.

```commandline
python -m example.1f_squidpy_analysis
```
This file performs an optional analysis using functions from `src.squidpy_stats`.


### Example 2: Testing with simulated data
Below we test simulated data from Sakura.
```commandline
python -m example.2a_creating_simulated_data
```
This file calls functions from `src.create` and creates an AnnData object with simulated expression, location, cell area, and labels.

```commandline
python -m example.2b_basic_spatial_analysis
```
This file exhibits a series of spatial analysis tools using functions from `src.spatial_stats` to help with spatial analysis.

```commandline
python -m example.2c_advanced_statistics
```
This file performs spatial enrichment and sensitivity analysis using functions from `src.enrichment`.


## License
Spatial Analysis Framework is licensed under the MIT License.

## Version
- **0.0.1** - First release.
