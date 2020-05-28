#!/usr/bin/env zsh

# split multiple series for the same person (manually detected through folder sizes and inspection of SliceLocation)
# run in ../../data/kaggle/raw/

s=test/test
i=1082
mv ${s}/${i} ${s}/${i}_b
mkdir -p ${s}/${i}_a/study 
mv ${s}/${i}_b/study/sax_10 ${s}/${i}_a/study
i=1111
mv ${s}/${i} ${s}/${i}_a
mkdir -p ${s}/${i}_b/study 
mv ${s}/${i}_a/study/sax_32 ${s}/${i}_b/study
i=721
mv ${s}/${i} ${s}/${i}_a
mkdir -p ${s}/${i}_b/study 
mv ${s}/${i}_a/study/sax_47 ${s}/${i}_b/study
i=793
mv ${s}/${i} ${s}/${i}_a
mkdir -p ${s}/${i}_b/study 
mv ${s}/${i}_a/study/sax_41 ${s}/${i}_b/study

s=train/train
i=123
mv ${s}/${i} ${s}/${i}_b
mkdir -p ${s}/${i}_a/study 
mv ${s}/${i}_b/study/sax_10 ${s}/${i}_a/study
i=334
mv ${s}/${i} ${s}/${i}_a
mkdir -p ${s}/${i}_b/study 
mv ${s}/${i}_a/study/sax_29 ${s}/${i}_b/study
# additional series detected through slice spacing: 437 (5-11 vs 12-22)
i=437
mv ${s}/${i} ${s}/${i}_a
mkdir -p ${s}/${i}_b/study 
mv ${s}/${i}_a/study/sax_{12..22} ${s}/${i}_b/study

s=validate/validate
i=516
mv ${s}/${i} ${s}/${i}_a
mkdir -p ${s}/${i}_b/study 
mv ${s}/${i}_a/study/sax_104 ${s}/${i}_b/study
i=619
mv ${s}/${i} ${s}/${i}_a
mkdir -p ${s}/${i}_b/study 
mv ${s}/${i}_a/study/sax_21 ${s}/${i}_b/study
