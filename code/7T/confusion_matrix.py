from glob import glob
import imageio
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
import re
import sys

if len(sys.argv) < 4:
    print("Usage: confusion_matrix.py <dir1> <dir2> <confusion.csv>")
    sys.exit(1)

def get_confusion(path1, path2):
    confusions = list()
    masks1 = sorted(glob("{}/**/*.png".format(path1)))
    masks2 = sorted(glob("{}/**/*.png".format(path2)))
    mask_dict1 = {re.sub("-.*.png","",Path(m).name): m for m in masks1}
    assert len(masks1) == len(mask_dict1)
    mask_dict2 = {re.sub("-.*.png","",Path(m).name): m for m in masks2}
    assert len(masks2) == len(mask_dict2)
    inter = mask_dict1.keys() & mask_dict2.keys()
    if(len(inter) != len(mask_dict1) or len(inter) != len(mask_dict2)):
        print("Different elements in lists: first: {}, second: {}, intersect: {}".format(len(mask_dict1),len(mask_dict2),len(inter)))
    for i in tqdm(inter):
        f1 = Path(mask_dict1[i])
        f2 = Path(mask_dict2[i])
        #assert f1.name == f2.name, "expected files to have the same name: {} - {}".format(f1.name, f2.name)
        m1 = imageio.imread(f1)
        m2 = imageio.imread(f2)
        # https://stackoverflow.com/a/50023660
        cm = np.zeros((3, 3), dtype=int)
        np.add.at(cm, (m1, m2), 1)
        confusions.append([i, f1.parent.name, f1.parent.parent.name, f2.parent.parent.name, *cm.flatten().tolist()])
    conf_table = pd.DataFrame(confusions, columns=["image","set","methodA","methodB","bg_bg","bg_lv","bg_my","lv_bg","lv_lv","lv_my","my_bg","my_lv","my_my"])
    return (conf_table)

conf = get_confusion(
    sys.argv[1],
    sys.argv[2]
).to_csv(sys.argv[3], sep="\t", index=False)
