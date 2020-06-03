# Knowledge Transfer and Transfer Learning for Cardiac Image Segmentation

This repository supplements the publication [link to be added], it contains the code, models and intermediate results to reproduce and further explore the presented results.
You can also use everything in this repository for your own research under [MIT License](./LICENSE). Please cite our publication if you do.
Generated data is available for download at [link to be added].

## Aim

This project aims to provide guidelines, code and data for researchers and clinicians who want to train their own cardiac MR segmentation network.

## Overview

 - Data Science Bowl Cardiac Challenge Data
   + Data curation and conversion
   + Creating labels with `ukbb_cardiac` by Bai et al.
   + Evaluating labels by ground truth volume info
   + Training own U-Nets on these labels - hyperparameter search
 - Transfer Learning: 7T human cardiac cine MRI
   + Comparsion of plain learning with transfer learning and double transfer learning
   + Assessing data requirements: training with subsets
   + Exploration of predictions

## Data Science Bowl Cardiac Challenge Data

The [Data Science Bowl Cardiac Challenge Data](https://www.kaggle.com/c/second-annual-data-science-bowl/data) (also "Kaggle data" for brevity) consists of cardiac MR scans of 1140 patients of different ages with different pathologies. It is publically available for download [at Kaggle](https://www.kaggle.com/c/second-annual-data-science-bowl/data).
Not much additional information is given regarding the scans but some metadata is stored in the DICOM files. The competition was to predict left ventricular (LV) volume of each patient at the end systolic (ES) and end diastolic (ED) phase. After conclusion of the competition the ground truth values for these volumes were released for all 1140 patients.

If you want to reproduce our results, download and unpack the kaggle images to `data/kaggle` (~74GB). You can do this using the web interface or the command line interface (in any case you need a Kaggle account and accept the competition rules):

```bash
mkdir -p data/kaggle/raw
cd data/kaggle/raw
kaggle competitions download second-annual-data-science-bowl
unzip second-annual-data-science-bowl.zip
rm second-annual-data-science-bowl.zip
```

To extract and analyze DICOM metadata use [dcm2metadata.py](./code/kaggle/dcm2metadata.py):
```bash
mkdir -p analysis/kaggle
cd analysis/kaggle
# this command takes ~30 minutes and (re-)creates dicom_metadata.tsv.xz
python ../../code/kaggle/dcm2metadata.py
```

### Data curation and conversion

The data set contains short axis images for all participants and has not been cleaned for the competition. Thus it contains inconsistencies:
 - missing time points (e.g. all slices with 30 cardiac phases and one with 25)
 - inconsistent slice spacing
 - inconsistent image dimension
 - repeat measurements (identical slice location)
 - scaled images
 - image rotations

Therefore we employ the following data cleaning steps:
1. Split completely repeated measurements (into _a and _b)
2. Downscale to 256px in the larger dimension
3. Scale and rotate to consistent size and orientation (using mode)
4. Rotate by 90° if `Rows<Columns` (also keep original rotation in a separate file)
5. Remove duplicate slices
6. Add missing slices (by duplicating existing ones)
7. Convert DICOM to nifti

```bash
# 1. split measurements
mkdir -p data/kaggle/nifti
cd data/kaggle/raw
../../../code/kaggle/kaggle_split_multiseries.sh
cd ../nifti
# 2.-7. this command takes ~2 hours and creates one sa.nii.gz file per patient in the kaggle dataset
python ../../../code/kaggle/dcm2nifti.py >../../../analysis/kaggle/conversion.log
```

These commands create one nifti file per patient in `data/kaggle/nifti` (plus a rotated one for patients where `Rows<Columns`).
Additionally these tables are created in `analysis/kaggle`:
 - [conversion.log](analysis/kaggle/conversion.log) - tab-separated info about conversion problems (columns: `pid,problem,resolution,image,extra_info`)
 - [used_dicoms.log](analysis/kaggle/used_dicoms.log) - tab-separated info about all dicom images used in the created niftis (columns: `pid,slice,frame,series,file_name`)
 - [patient_dimensions.tsv](analysis/kaggle/patient_dimensions.tsv) - tab-separated info about image dimensions by patient (columns: `pid,X,Y,pixelSpacingX,pixelSpacingY,sliceSpacing,scaleFactor`)

### Gather metadata and ground truth by patient

```
mkdir -p analysis/kaggle/truth
cp data/kaggle/raw/{solution,train,validate}.csv analysis/kaggle/truth
# This command creates the two files: combined_metadata.csv and patient_metadata.csv
Rscript code/kaggle/kaggle_metadata.r
```

Now you can interactively explore the metadata in this notebook (requires an R kernel): [kaggle_metadata.ipynb](code/kaggle/kaggle_metadata.ipynb)

### Creating labels with `ukbb_cardiac` by Bai et al.

As no ground truth segmentation labels are available we automatically generate them using the network published by Bai et al.:

> W. Bai, et al. Automated cardiovascular magnetic resonance image analysis with fully convolutional networks. Journal of Cardiovascular Magnetic Resonance, 20:65, 2018. https://doi.org/10.1186/s12968-018-0471-x

Source code is available at: https://github.com/baiwenjia/ukbb_cardiac under Apache-2.0 license.

```
# download ukbb_cardiac (from fork so ED frame is determined by volume rather than assumed at frame 1 and for compatibility with tensorflow v2)
git clone https://github.com/chfc-cmi/ukbb_cardiac ~/ukbb_cardiac
# make sure to install all dependencies, I recommend a separate conda environment
cd ~/ukbb_cardiac
# to test that everything is working and to download the pre-trained models
python demo_pipeline.py
# cd in the cmr-seg-tl/data/kaggle/nifti directory
python ~/ukbb_cardiac/common/deploy_network.py --seq_name sa --data_dir . --model_path ~/ukbb_cardiac/trained_model/FCN_sa
python ~/ukbb_cardiac/short_axis/eval_ventricular_volume.py --data_dir . --output_csv ../../../analysis/kaggle/ukbb_ventricular_volumes.csv
```

This creates a prediction file `seg_sa.nii.gz` for each patient in the data subfolders. In addition files and predictions for the end systolic (`sa_ES.nii.gz`, `seg_sa_ES.nii.gz`) and end diastolic phase (`sa_ED.nii.gz`, `seg_sa_ED.nii.gz`) are created. Volumes are automatically derived with the last command and stored in [`ukbb_ventricular_volumes.csv`](analysis/kaggle/ukbb_ventricular_volumes.csv).

### Evaluating labels by ground truth volume info

The `ukbb_cardiac` network by Bai et al. is trained on homogenous UK Biobank data and not expected to perform well on every patient of the heterogeneous Kaggle data.

### Training own U-Nets on these labels - hyperparameter search
TODO

## Transfer Learning: 7T human cardiac cine MRI
TODO

### Comparsion of plain learning with transfer learning and double transfer learning
TODO

### Assessing data requirements: training with subsets
TODO

### Exploration of predictions
TODO

## Requirements
A list of required programs and packages. The listed version is the one used for our analysis (older or newer versions might work but are untested).

### Hardware
In order to reproduce our results in an acceptable time frame GPU hardware is required. The setup we used was:
1)	GPU Node of Julia HPC of University of Wuerzburg:
i.	2x Intel® Xeon Gold 6134 Processor
ii.	384 GB of memory
iii.	2x NVIDIA Tesla P100 with 16 GB of memory
2)	Our own HPC within the Comprehensive Heart Failure Center
i.	8x Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz
ii.	512 GB of memory
iii.	1x NVIDIA Tesla K80 with 12 GB of memory
All calculations on data underlying the data protection terms of the University Hospital Wuerzburg were done on our own HPC, while all calculations on publicly available data, such as the parameter search on Kaggle data were done on the Julia HPC of the University of Wuerzburg.
### Software

#### Python
 - python 3.8.3
 - jupyter 1.0.0
 - pydicom 1.4.2
 - nibabel 3.1.0
 - pandas 1.0.3
 - numpy 1.13.3
 - scipy 1.4.1
 - scikit-image 0.16.2
 - tqdm 4.46.0
 - ukbb_cardiac v2.0 (dependencies incl. tensorflow 2.1.0 installed in a separate conda env)

#### R
 - R 3.6.1
 - tidyverse 1.2.1