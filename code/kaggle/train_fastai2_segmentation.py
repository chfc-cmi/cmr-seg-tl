#!/usr/bin/env python
# coding: utf-8

from fastai.vision.all import *
import pandas as pd

imgList = pd.read_csv("../../analysis/kaggle/image_list_filtered_score.tsv", sep="\t")
filteredList = imgList[imgList.score<=.05]

heart = DataBlock(blocks=(ImageBlock, MaskBlock(codes = np.array(["background","left_ventricle","myocardium"]))),
    get_x=ColReader('file', pref="../../data/data/"),
    splitter=ColSplitter('is_val'),
    get_y=lambda x: str(x['file']).replace("images", "../../data/data/masks_2class"),
    item_tfms=Resize(256, method='pad', pad_mode='zeros'),
    batch_tfms=aug_transforms(do_flip=True,max_rotate=90,max_lighting=.4,max_zoom=1.2))

dls = heart.dataloaders(filteredList, bs=32)

def acc_seg(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

def multi_dice(input:Tensor, targs:Tensor, class_id=0, inverse=False):
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    # replace all with class_id with 1 all else with 0 to have binary case
    output = (input == class_id).float()
    # same for targs
    targs = (targs.view(n,-1) == class_id).float()
    if inverse:
        output = 1 - output
        targs = 1 - targs
    intersect = (output * targs).sum(dim=1).float()
    union = (output+targs).sum(dim=1).float()
    res = 2. * intersect / union
    res[torch.isnan(res)] = 1
    return res.mean()

def diceComb(input:Tensor, targs:Tensor):
    return multi_dice(input, targs, class_id=0, inverse=True)
def diceLV(input:Tensor, targs:Tensor):
    return multi_dice(input, targs, class_id=1)
def diceMY(input:Tensor, targs:Tensor):
    return multi_dice(input, targs, class_id=2)

learn = unet_learner(dls, resnet34, metrics=[acc_seg,diceComb,diceLV,diceMY], cbs=[CSVLogger(append=True)], path='fastai2')

lr=1e-4
learn.freeze()
learn.fit_one_cycle(30, lr)
learn.unfreeze()
lr=1e-5
learn.fit_one_cycle(30, lr)

learn.export('model.pkl')
