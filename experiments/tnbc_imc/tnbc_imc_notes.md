## TNBC IMC analysis
______________________________
## Steps for preprocessing and analysis

### Setup and record
   - `tnbc_xp_setup.py`: TNBC TMA and experiments information, including core-roi information.
   - `tnbc_record_manual_annot.py`: record of manual annotation for clusters and cell types.


### Prepare/organize the data
   - `tnbc_tma_1_batch_effect.py`: Organize dataset and check batch effects between different experiments.
   - `.py`:


### Classification of ESI cell types
   - `tnbc_tma_2_GMM.py`: classify epithelial and non-epithelial cells
   - `tnbc_tma_3_annot.py`: update and merge ESIU labels by manual annotations.
   - `.py`: 


