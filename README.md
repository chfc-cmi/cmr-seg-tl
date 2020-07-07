# Knowledge Transfer and Transfer Learning for Cardiac Image Segmentation

This repository supplements the publication [*A Deep Learning Based Cardiac Cine Segmentation Framework for Clinicians - Transfer Learning Application to 7T*](https://doi.org/10.1101/2020.06.15.20131656 ), it contains the code, models and intermediate results to reproduce and further explore the presented results.
You can also use everything in this repository for your own research under [MIT License](./LICENSE). Please cite our publication if you do:
>  A Deep Learning Based Cardiac Cine Segmentation Framework for Clinicians - Transfer Learning Application to 7T
Markus J Ankenbrand, David Lohr, Wiebke Schlötelburg, Theresa Reiter, Tobias Wech, Laura Maria Schreiber
medRxiv 2020.06.15.20131656; doi: https://doi.org/10.1101/2020.06.15.20131656 

```BibTeX
@article {Ankenbrand2020.06.15.20131656,
	author = {Ankenbrand, Markus J and Lohr, David and Schl{\"o}telburg, Wiebke and Reiter, Theresa and Wech, Tobias and Schreiber, Laura Maria},
	title = {A Deep Learning Based Cardiac Cine Segmentation Framework for Clinicians - Transfer Learning Application to 7T},
	elocation-id = {2020.06.15.20131656},
	year = {2020},
	doi = {10.1101/2020.06.15.20131656},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2020/06/17/2020.06.15.20131656},
	eprint = {https://www.medrxiv.org/content/early/2020/06/17/2020.06.15.20131656.full.pdf},
	journal = {medRxiv}
}
```

**Note:** This README reports all the commands necessary to reproduce our results from the raw data. However, if you want to use some intermediate files you do not need to follow all steps but you can find the precalculated data either in this repository or at zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3876351.svg)](https://doi.org/10.5281/zenodo.3876351).

<!-- TOC -->

- [Knowledge Transfer and Transfer Learning for Cardiac Image Segmentation](#knowledge-transfer-and-transfer-learning-for-cardiac-image-segmentation)
    - [Aim](#aim)
    - [Overview](#overview)
    - [Data Science Bowl Cardiac Challenge Data](#data-science-bowl-cardiac-challenge-data)
        - [Data curation and conversion](#data-curation-and-conversion)
        - [Gather metadata and ground truth by patient](#gather-metadata-and-ground-truth-by-patient)
        - [Creating labels with `ukbb_cardiac` by Bai et al.](#creating-labels-with-ukbb_cardiac-by-bai-et-al)
        - [Evaluating labels by ground truth volume info](#evaluating-labels-by-ground-truth-volume-info)
        - [Training own U-Nets on these labels - hyperparameter search](#training-own-u-nets-on-these-labels---hyperparameter-search)
    - [Transfer Learning: 7T human cardiac cine MRI](#transfer-learning-7t-human-cardiac-cine-mri)
        - [Data preparation](#data-preparation)
            - [Images](#images)
            - [Masks](#masks)
            - [Split into training, validation and test set](#split-into-training-validation-and-test-set)
        - [Train networks with plain learning, transfer learning, and double transfer learning](#train-networks-with-plain-learning-transfer-learning-and-double-transfer-learning)
        - [Assessing data requirements: training with subsets](#assessing-data-requirements-training-with-subsets)
        - [Exploration of predictions](#exploration-of-predictions)
            - [Analysis notebooks](#analysis-notebooks)
    - [Requirements](#requirements)
        - [Hardware](#hardware)
        - [Software](#software)
            - [Python](#python)
            - [R](#r)
            - [Other](#other)

<!-- /TOC -->

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

Detailed evaluations and figures are in this notebook: [`code/kaggle/compare_predictions.ipynb`](./code/kaggle/compare_predictions.ipynb). An important result from that evaluation is this table of absolute differences in ejection fraction (EF) between predictions and truth:

| method | mean | sd | median | iqr |
|--------|-----:|---:|-------:|----:|
| r34_p05_s256 | 3.64 | 3.38 | 2.87 | 3.72 |
| r50_p05_s256 | 3.71|3.73|2.79|3.70|
| r34_p15_s256 | 3.73|3.38|2.91|3.72|
| r34_p05_s256_focal | 3.75|3.90|2.86|3.80|
| r34_p10_s256 | 3.77|4.44|2.89|3.76|
| r34_p05_s256_extremeTfms | 3.78|3.59|2.89|3.64|
| r34_p05_s128 | 3.90|4.26|3.05|3.86|
| r34_p05_s256_dice | 3.93|3.43|3.07|3.91|
| v16_p05_s256 | 4.06|4.94|3.02|3.87|
| UKBB | 5.42|8.83|3.72|4.34|

The resnet34 backbone on the 5% confidence set, image size 256, cross entropy loass and normal transformation has lowest mean absolute difference.  When applied on the images with `Rows<Columns` (unrotated) however, the model with extreme transformations clearly outperforms all other models.

The final model with extreme transforms was uploaded as part of the preliminary release as it is too large to be included in the repository: https://github.com/chfc-cmi/cmr-seg-tl/releases/download/v0.1.0/resnet34_5percent_size256_extremeTfms_ceLoss.pkl

You can get the pretrained model as a fastai (v1) learner with a single command using torch hub:

```python
model = torch.hub.load("chfc-cmi/cmr-seg-tl","cmr_seg_base")
# alternatively download the pkl file and load it manually
model = fastai.vision.load_learner(path=".", file="resnet34_5percent_size256_extremeTfms_ceLoss.pkl", tfm_y=False)
```

## Transfer Learning: 7T human cardiac cine MRI

This section describes all the computational steps required to perform transfer learning on your own data. In our case it was CINE images acquired at 7T. Ethics approval for this study was granted by the local ethics commitee (University Hospital Würzburg, 7/17-SC) and written informed consent from all participants was obtained. Unfortunately, we are not allowed to share the acquired images data due to data protection and privacy regulations.

### Data preparation

#### Images

All dicom files for the short axis cine are stored in `data/7T/dicom` with sub directories for each volunteer. These dicom files were converted into consistently named png files using this code:

```bash
# in data/7T/images
bash ../../../code/7T/dcm2png.sh
```

#### Masks

The manual segmentation was performed using `QMass` from the [Medis Suite](https://medisimaging.com/apps/lv-rv-function/) (version TODO). The endocardial and epicardial contours were drawn for all slices and timepoints. For each volunteer the contours were then exported as one `.con` file.

To convert the `.con` files to `.png` images the following steps were applied with all `.con` files in `data/7T/contours/original`:

```bash
# in data/7T/contours
bash ../../../code/7T/con_to_png.sh
```

This converts each con file into a tsv and then creates png files from these. It also creates a `resolution.tsv` file and black png files with proper dimensions. These are used to fill slices and timepoints with empty masks if they are not part of the con files:

```bash
ls masks | grep "\-mask" | cut -f1 -d"-" >masklist
ls images | cut -f1 -d"-" >imagelist

for i in $(cat imagelist masklist | sort | uniq -u)
do
    COL=$(grep -w $(echo $i | cut -c1,2) contours/resolution.tsv | cut -f2)
    cp masks/empty_${COL}x*.png masks/$i-mask.png
done

rm masklist imagelist
```

In our case a second expert labeled some of the volunteers again, independently. These labels are used to assess interobserver variability. The same steps for conversion from con files to png need to be repeated.

#### Split into training, validation and test set

The test set was chosen as those volunteers labeled by both experts (ids 26, 27 and 28). Five of the remaining 19 volunteers were selected for the validation set, randomly:

```bash
# in data/7T
mkdir -p {masks,images}/{train,test,val}
for i in images masks
do
	mv $i/{26,27,28}_*.png $i/test
	mv $i/{30,36,19,42,35}_*.png $i/val
	mv $i/*-*.png $i/train
done
```

A table with pixel counts per class [pixel_counts_by_class.tsv](./analysis/7T/pixel_counts_by_class.tsv) can be generated with:

```bash
python code/7T/count_classes_in_masks.py
```

This can be used to determine end-systolic and end-diastolic phases.
To further sub-divide the training set into smaller sub-sets for data requirement determination run this command:

```bash
Rscript code/7T/create_subset_table.r
```

it creates the two files [esed_frames.tsv](./analysis/7T/esed_frames.tsv) and [image_subsets.tsv](./analysis/7T/image_subsets.tsv) in `analysis/7T`.

### Train networks with plain learning, transfer learning, and double transfer learning

Use [this notebook](./code/7T/transfer_learning.ipynb) to train three models on this data:
 - a plain model with random weights (further referred to as `R`)
 - a model with backbone weights from ImageNet pre-training (further referred to as `TL`)
 - the model we trained on the kaggle data (`r34_p05_s256` referred to as `base`) (further referred to as `TL2`)

In addition to the models, also predictions on all images for all models (`R`, `TL`, `TL2` and `base`) are generated as images (raw: as probabilities in the color channels and discretized as masks).

### Assessing data requirements: training with subsets

With the [image_subsets.tsv](./analysis/7T/image_subsets.tsv) file in the same folder as the images and masks you can run training with the defined subsets like this:

```bash
python segmentation_with_reduced_set.py --set v7                  
python segmentation_with_reduced_set.py --set r7                              
python segmentation_with_reduced_set.py --set v3
python segmentation_with_reduced_set.py --set r3
python segmentation_with_reduced_set.py --set v1
python segmentation_with_reduced_set.py --set r1
python segmentation_with_reduced_set.py --set esed
python segmentation_with_reduced_set.py --set r_esed
# make full predictions on all images
python full_prediction_reduced_sets.py
```

Again in addition to the models, predictions on all images for all models are generated in accordingly named subfolders of `preds`. Furthermore, logs of the training process with validation performance after each epoch are written.
These are collected into one tsv file using this command:

```bash
# in data/7T/images
echo epoch,train_loss,valid_loss,acc_seg,diceComb,diceLV,diceMY,time,model >../../../analysis/7T/train_log_combined.csv
for i in *_log.csv
do
  perl -ne 'BEGIN{$l=0}chomp;s/^\d+,/$l,/;unless(/n/){print "$_,'$(basename $i _log.csv)'\n";$l++}' $i >>../../../analysis/7T/train_log_combined.csv
done
```

### Exploration of predictions

For detailed exploration and comparison of predictions (with pixel resolution) consistently downscaled versions of the original images and masks as well as full predictions using `ukbb_cardiac` on these are generated with:

```bash
python code/7T/rescale_images_masks.py
python code/7T/predict_pngs_ukbbCardiac.py
```

The consistently scaled masks (manually created and predicted using the different models) can be used to calculate pair-wise confusion matrices (per-image):

```bash
CODE=../../code/7T
OUT=../../analysis/7T/confusion_tables
mkdir -p $OUT
# in data/7T
python $CODE/confusion_matrix.py scaled_masks preds/plainLearn $OUT/confusion_WS_plainLearn.tsv
python $CODE/confusion_matrix.py scaled_masks preds/imagenetTransferLearn $OUT/confusion_WS_imagenetTransferLearn.tsv
python $CODE/confusion_matrix.py scaled_masks preds/baseModel $OUT/confusion_WS_base.tsv
python $CODE/confusion_matrix.py scaled_masks preds/doubleTransferLearn $OUT/confusion_WS_doubleTransferLearn.tsv
python $CODE/confusion_matrix.py scaled_masks preds/doubleTransferLearn_esed $OUT/confusion_WS_doubleTL-esed.tsv
python $CODE/confusion_matrix.py scaled_masks scaled_masks_TR $OUT/confusion_WS_TR.tsv
python $CODE/confusion_matrix.py scaled_masks ukbb_preds_3class $OUT/confusion_WS_ukbbCardiac.tsv
python $CODE/confusion_matrix.py scaled_masks_TR preds/doubleTransferLearn $OUT/confusion_TR_doubleTransferLearn.tsv
```

These confusion tables are included in the repository.

#### Analysis notebooks

These notebooks can be used to analyse the data. Except for the last one they only use the derived data included in the repository. So you can use them to reproduce our results and analyze them in more detail. The last notebook can be used with your own data to inspect specific predictions in comparison to ground truth:

- [compare_predictions.ipynb](./code/7T/compare_predictions.ipynb) - analysis of the image-wise predictions of the different models
- [tl_performance.ipynb](./code/7T/tl_performance.ipynb) - analysis of performance throughout training on the validation set, comparison of subsets
- [image_overlay.ipynb](./code/7T/image_overlay.ipynb) - overlay of predictions on images and ground truth

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
 - pytorch 1.4.0
 - fastai 1.0.60
 - seaborn 0.10.1

#### R
 - R 3.6.1
 - tidyverse 1.2.1
 - patchwork 1.0.0
 - ggfortify 0.4.10

#### Other
 - [med2image](https://github.com/FNNDSC/med2image) 2.0.1
 - [mitools](http://neuro.debian.net/pkgs/mitools.html) 2.0.3 (miconv)
 - dos2unix 7.4.0
