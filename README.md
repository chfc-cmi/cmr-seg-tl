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

To extract and analyse DICOM metadata use: TODO

### Data curation and conversion

The data set contains short axis images for all participants and has not been cleaned for the competition. Thus it contains inconsistencies:
 - missing time points (e.g. all slices with 30 cardiac phases and one with 25)
 - inconsistent slice spacing
 - inconsistent image dimension
 - repeat measurements (identical slice location)
 - scaled images
 - image rotations

Therefore we employ the following data cleaning steps:
1. Split completely repeated measurements
2. TODO

### Creating labels with `ukbb_cardiac` by Bai et al.

As no ground truth segmentation labels are available we automatically generated them using the network published by Bai et al. (TODO cite).

### Evaluating labels by ground truth volume info
TODO

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

