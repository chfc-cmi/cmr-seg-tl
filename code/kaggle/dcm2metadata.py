#!/usr/bin/env python
# coding: utf-8

import pydicom as dicom
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm

# adjusted from https://github.com/pydicom/pydicom/issues/319#issuecomment-282954803
def basedict(ds):
    """Turn a pydicom Dataset into a dict with keys derived from the Element tags.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The Dataset to dictify

    Returns
    -------
    output : dict
    """
    output = dict()
    for elem in ds:
        key = elem.name
        if elem.tag.is_private:
            key = elem.name + str(elem.tag).replace(", ","")
        # skip Pixel Data, CSA Series Header Info, CSA Image Header Info (potentially others)
        if elem.VR.startswith("O"):
            continue
        if elem.VR != 'SQ':
            output[key] = elem.value
        else:
            output[key] = [basedict(item) for item in elem]
    return output

files = glob("../../data/kaggle/raw/*/*/*/study/sax_*/*.dcm")

metadata = list()
for f in tqdm(files):
    d = dicom.read_file(f)
    fd = basedict(d)
    f = Path(f)
    fd["file"] = str(f.name)
    fd["dir"] = str(f.parent.name)
    fd["pid"] = str(f.parent.parent.parent.name)
    metadata.append(fd)

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv("../data/kaggle-heart/nifti/dicom_metadata.tsv.xz",sep="\t",index=False)
