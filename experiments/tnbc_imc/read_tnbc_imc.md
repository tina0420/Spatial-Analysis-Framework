## TNBC IMC spatial analysis
______________________________

## Steps for data preprocessing and downstream analysis
### Transferring .MCD to .tiff
Use IMC Converter to get ROIs from each slide.

### Setup and record
   - All the .MCD from wet lab are under the directory: `data/tnbc_imc`.
   - SectorMaps of all TMAs are stored in `tnbc_xp_setup.py`.
   - Markers related information: `tnbc_data_info.py`
   - IMC analysis specifically setup is here: `tnbc_imc_analysis_setup.py`.
   - If you want to add some information into the setup file, here are some general setup process: `tnbc_general_setup.py`.
   - Record of all manual annotation for clusters and cell types as a backup: `tnbc_record_manual_annot.py`.


### Single-cell Segmentation and Quantification of the Marker Expression
   - `tnbc_imc_mesmer.py`: Segmenting single cells in each ROI by Mesmer (CD68, SMA, CD45, CD31 and panCK were used as cytoplasm).
   - `tnbc_imc_measurement_quantify.py`: Quantify marker expressions, normalize them and store them in Anndata format.



### Marker Quality Control pipeline for Spatial Dataset
   - `tnbc_imc_marker_quality_control.py`: Running marker quality control pipeline.
   - `qc_function_list.md`: Functions used in the marker quality control pipeline.



### Extract useful markers for each experiments and check batch effect
- `tnbc_data_info.py`: With all the experiment names and their corresponding marker quality results.
- `tnbc_rsi.py`: Indentify epithelial, immune and stromal cells.



### Checking batch effect between experiments and TMAs
    - `tnbc_tma_1_batch_effect.py`: .
  - `tnbc_tma_2_gmm.py`: classify epithelial and non-epithelial cells by 2-component Gaussian mixture model.
  - `tnbc_tma_3_leiden.py`: Running Leiden Clustering for identifying immune and stromal cell types from TME.
  - ``: .



#### Some specific experiments for specific focus
   - `tnbc_imc_xp_batch.py`: .
   - `tnbc_imc_tma_batch.py`: .
   - `tnbc_imc_cohort_analysis_prepare.py`: Prepare all the data and information needed for cohort analysis.



### Patient information/metadata organization
   - `tnbc_imc_metadata.py`: .
   - `tnbc_imc_cohort_organization.py`: Organizing cell data and patient information of the ctDNA TNBC cohort.
   - ``



### Cell type annotations for ctDNA TNBC cohort (TNBC IMC by BM+ panel, 4 TMAs)
   - `tnbc_imc_filter_good_data.py`: Filtering out the markers marked as failed or unclear by pathologists(stored in `tnbc_data_info.py`).
   - `tnbc_imc_layer1_gmm.py`: Running 2-component GMM for identifying epithelial and TME cells in layer 1.
   - `tnbc_imc_layer1_tme_leiden.py`: Running leiden clustering with TME cells and annotate immune and stromal cells in layer 1.
   - `tnbc_imc_layer1_merge.py`: Merge all layer 1 (ESI) cell types together (from epi and tme).
   - `tnbc_imc_layer2_epi_leiden.py`: Running leiden clustering with all epithelial cells in layer 1.
   - `tnbc_imc_layer2_tme_leiden.py`: Annotating layer 2 cell types of tme.
   - `tnbc_imc_layer2_merge.py`: Merge all layer 2 cell types together (from epi and tme).



### Downstream analysis
   - `tnbc_imc_multiple_cores_per_pid.py`: Checking the images by patients to decide which images to use.
   - `tnbc_imc_cell_type_abundance.py`: Cell type proportions calculated for each patient/image.
   - `tnbc_imc_cliff_delta.py`: Compute and plot cliff delta.
   - `tnbc_imc_tils.py:`


    # # Choose the first image if a patient has more than one image
    for pid in 
    # patient_obs = adata[adata.obs[patient_key] == pid].obs[[cluster_key, observation_key]]
    # # print(patient_obs)
    # # Choose the first image from the patient as one image is strong to represent (citation)
    # patient_obs = patient_obs[patient_obs[img_key] == list(set(patient_obs[img_key]))[0]]





