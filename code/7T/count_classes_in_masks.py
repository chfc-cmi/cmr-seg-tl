#!/usr/bin/env python
# coding: utf-8

from glob import glob
import imageio
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path


def count_pixels_by_class(path):
    counts = list()
    masks = sorted(glob("{}/**/*.png".format(path),recursive=True))
    for i in tqdm(masks):
        p = Path(i)
        m = imageio.imread(i)
        # https://stackoverflow.com/a/50023660
        cm = np.zeros((3,), dtype=int)
        np.add.at(cm, (m, ), 1)
        counts.append([p.parent.parent.name, p.parent.name, p.name, *cm.flatten().tolist()])
    return (counts)


counts = count_pixels_by_class('data/7T/masks')
counts_df = pd.DataFrame(counts, columns=["source","set","file","bg","lv","my"])
counts_df.to_csv("analysis/7T/pixel_counts_by_class.tsv", sep="\t", index=False)