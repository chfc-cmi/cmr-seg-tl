# Knowledge Transfer and Transfer Learning for Cardiac Image Segmentation

This repository supplements the publication [link to be added], it contains the code, models and intermediate results to reproduce and further explore the presented results.
You can also use everything in this repository for your own research under [MIT License](./LICENSE). Please cite our publication if you do.

**Note:** This README reports all the commands necessary to reproduce our results from the raw data. However, if you want to use some intermediate files you do not need to follow all steps but you can find the precalculated data either in this repository or at zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3876351.svg)](https://doi.org/10.5281/zenodo.3876351).

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

```bash
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

```bash
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

In order to easily use the data with fastai the images and corresponding created masks need to be converted to png (in `data/kaggle`). This will take some time and result in three sub folders (`images`, `masks` and `masks_2class`):

```bash
# in data/kaggle
mkdir -p masks masks_2class images
for i in nifti/*/seg_sa.nii.gz
do
    BASE=$(basename $(dirname $i))
    mkdir -p masks/$BASE masks_2class/$BASE
    echo $i
    miconv -noscale -pflip $i masks/$BASE/$BASE.png
done

find masks | xargs rename 's/_time/-frame0/;s/_slice(\d+)/sprintf "-slice%03d", $1/e'
python ../../code/kaggle/from3to2.py

for i in nifti/*/sa.nii.gz
do
    BASE=$(basename $(dirname $i))
    echo $i
    med2image -i $i -d images/$BASE -o $BASE.png -t png
done

find images -name "*.png" | perl -F"/" -ape '$_="$F[1]\t$_"' | xz >../../analysis/kaggle/image_list.tsv.xz
```

Two different conversion tools are used for images and masks. The reason is that `med2image` does a good job scaling the grey range but there is no way to disable normalization (like the `-noscale` option in `miconv`) which results in erroneous grey values for masks that lack some classes.

The intermediate step of renaming is to match the filenames of the images and masks. We produce dedicated images with just two (non-background) classes (left ventricle and myocardium) while marking the right ventricle as background. This helps avoid a pre-processing step in further steps as we only want to train models on this two class setting.

The last command creates a list of all image files with patient id.

These images and masks are available for download together with the confidence values created in the next section at zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3876351.svg)](https://doi.org/10.5281/zenodo.3876351)



### Evaluating labels by ground truth volume info

The `ukbb_cardiac` network by Bai et al. is trained on homogenous UK Biobank data and not expected to perform well on every patient of the heterogeneous Kaggle data.

We use the ground truth data for the end systolic and end diastolic left ventricular volume as provided by Kaggle to estimate the confidence in the masks of each patient. It is important to note that this estimation solely relies on the LV class at two distinct timepoints. Also the volume information alone does not guarantee segmentation of the correct structure. Still, limited visual inspection showed good correspondence between these confidence values and qualitative segmentation performance.

We define the confidence value as the larger of the absolute deviations between calculated and true volume at end diastole and end systole in percent.

A table with confidence values can be created with this command (in `analysis/kaggle`):

```bash
Rscript ../../code/kaggle/get_confidence.r
```

This will create the two files `confidence_by_patient.tsv` and `image_list_filtered_score.tsv` which are also included in the zenodo archive.

### Training own U-Nets on these labels - hyperparameter search

Different U-Nets were trained using the `fastai` framework and a range of parameters with regard to:
 - backbone architecture
 - data augmentations
 - image size
 - input data (confidence sets)

In order to run the following commands (in a sensible time frame) you need access to a machine with a GPU unit. Put your kaggle `images` and `masks_2class` as well as the `image_list.tsv.xz` and `image_list_filtered_score.tsv` in one folder on that machine. There you can run this script with the parameters of your choice. These were the experiments we tried:

```bash
python train_fastai_segmentation.py --size 256 --bs 32 --confidence 5 --model resnet34 --tfms normal --loss ce
# Try different models
python train_fastai_segmentation.py --size 256 --bs 8 --confidence 5 --model resnet50 --tfms normal --loss ce
python train_fastai_segmentation.py --size 256 --bs 8 --confidence 5 --model vgg16 --tfms normal --loss ce
# Try different loss functions
python train_fastai_segmentation.py --size 256 --bs 32 --confidence 5 --model resnet34 --tfms normal --loss focal
python train_fastai_segmentation.py --size 256 --bs 32 --confidence 5 --model resnet34 --tfms normal --loss softdice
# Try different confidence sets
python train_fastai_segmentation.py --size 256 --bs 32 --confidence 10 --model resnet34 --tfms normal --loss ce
python train_fastai_segmentation.py --size 256 --bs 32 --confidence 15 --model resnet34 --tfms normal --loss ce
# Try different image size
python train_fastai_segmentation.py --size 128 --bs 32 --confidence 5 --model resnet34 --tfms normal --loss ce
# Try different augmentation
python train_fastai_segmentation.py --size 256 --bs 32 --confidence 5 --model resnet34 --tfms extreme --loss ce
```

This creates one model per run (+checkpoints) and a `predictions.tsv` file with predicted number of LV and MY pixels for all images.

The `predictions.tsv` files we generated are in `analysis/kaggle/predictions`.

TODO evaluation notebooks with figures

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

#### Other
 - [med2image](https://github.com/FNNDSC/med2image) 2.0.1
 - [mitools](http://neuro.debian.net/pkgs/mitools.html) 2.0.3 (miconv)