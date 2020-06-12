#!/usr/bin/env python
# coding: utf-8

import argparse
from fastai.vision import *
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import sys

from fastai.callbacks import CSVLogger

# suppress anoying and irrelevant warning, see https://forums.fast.ai/t/warnings-when-trying-to-make-an-imagedatabunch/56323/9
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

parser = argparse.ArgumentParser()
parser.add_argument('--size', dest='size', help='scale images to size', default=256, type=int)
parser.add_argument('--bs', dest='bs', help='batch size', default=32, type=int)
parser.add_argument('--cuda_device', dest='cuda_device', help='cuda device index', default=0, type=int)
parser.add_argument('--confidence', dest='confidence', help='confidence cutoff in percent', default=10, type=int)
parser.add_argument('--model', dest='model', help='model, one of resnet34, resnet50, vgg16', default='resnet34', type=str)
parser.add_argument('--tfms', dest='tfms', help='transformations, one of the presets no, normal, extreme', default='normal', type=str)
parser.add_argument('--loss', dest='loss', help='loss function, one of the presets ce, focal, softdice', default='ce', type=str)
args = parser.parse_args()

our_models = {"resnet34": models.resnet34, "resnet50": models.resnet50, "vgg16": models.vgg16_bn}
our_tfms = {
    "no": None,
    "normal": get_transforms(do_flip=False,max_rotate=20,max_lighting=.4,max_zoom=1.2),
    "extreme": get_transforms(do_flip=True,max_rotate=90,max_lighting=.4,max_zoom=1.2)
}

if args.loss not in ["ce", "focal", "softdice"]:
    sys.exit("Unknown loss function") 
size = args.size
bs = args.bs
cuda_device = args.cuda_device
confidence_cutoff = args.confidence/100
model = our_models[args.model]
tfms = our_tfms[args.tfms]
name = "noPretrain_{}_{}percent_size{}_{}Tfms_{}Loss".format(args.model,args.confidence,size,args.tfms,args.loss)

torch.cuda.set_device(cuda_device)

os.mkdir(name)

get_y_fn = lambda x: str(x).replace("images", "masks_2class")

imgList = pd.read_csv("nifti/image_list_filtered_score.tsv", sep="\t")
filteredList = imgList[imgList.score<=confidence_cutoff]

src = (SegmentationItemList.from_df(filteredList,path="nifti",cols="file")
       .split_from_df(col='is_val')
       .label_from_func(get_y_fn, classes=np.array(["background","left_ventricle","myocardium"])))

data = (src.transform(tfms,size=size,padding_mode="zeros",resize_method=ResizeMethod.PAD,tfm_y=True)
       .databunch(bs=bs)
       .normalize(imagenet_stats))

def acc_seg(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

def multi_dice(input:Tensor, targs:Tensor, class_id=0, inverse=False)->Rank0Tensor:
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

dice0inv = partial(multi_dice, class_id=0, inverse=True)
dice1 = partial(multi_dice, class_id=1)
dice2 = partial(multi_dice, class_id=2)
dice0inv.__name__ = 'diceComb'
dice1.__name__ = 'diceLV'
dice2.__name__ = 'diceMY'

class SoftDiceLoss(nn.Module):
    ''' 
    WARNING: this implementation does not work in our case, assumes one hot and channel last - need to restructure or re-write
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
    # Arguments
        targets: b x X x Y( x Z...) x c One hot encoding of ground truth
        inputs: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    # References
        https://www.jeremyjordan.me/semantic-segmentation/ (https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py)
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # skip the batch and class axis for calculating Dice score
        print(inputs.shape)
        print(inputs)
        print(targets.shape)
        print(targets)
        axes = tuple(range(1, len(inputs.shape)-1)) 
        numerator = 2. * np.sum(inputs * targets, axes)
        denominator = np.sum(np.square(inputs) + np.square(targets), axes)
        return 1 - np.mean(numerator / (denominator + self.epsilon)) # average over classes and batch

# adjusted from https://forums.fast.ai/t/loss-function-of-unet-learner-flattenedloss-of-crossentropyloss/51605
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = CrossEntropyFlat(axis=1,reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = ((1 - pt)**self.gamma) * CE_loss
        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()

learn = unet_learner(data, model, pretrained=False, metrics=[acc_seg,dice0inv,dice1,dice2], callback_fns=[partial(CSVLogger, append=True, filename="train_log")], path=name)

if args.loss == "focal":
    learn.loss_func = FocalLoss()
if args.loss == "softdice":
    learn.loss_func = SoftDiceLoss()

learn.unfreeze()

lr_find(learn)
fig = learn.recorder.plot(return_fig=True)
fig.savefig(name+"/lrfind_unfreeze.png")

lr=1e-5
learn.fit_one_cycle(5, lr)
learn.save(name+'-unfreeze-5')
fig = learn.recorder.plot_losses(return_fig=True)
fig.savefig(name+"/unfreeze-5.png")

learn.fit_one_cycle(10, lr)
learn.save(name+'-unfreeze-15')
fig = learn.recorder.plot_losses(return_fig=True)
fig.savefig(name+"/unfreeze-15.png")

learn.fit_one_cycle(15, lr)
learn.save(name+'-unfreeze-30')
fig = learn.recorder.plot_losses(return_fig=True)
fig.savefig(name+"/unfreeze-30.png")

learn.export('model.pkl')

# Make Predictions

fullImgList = pd.read_csv("nifti/image_list.tsv", sep="\t", header=None, names=["pid","file"])

pixelTable = pd.DataFrame({'file': [], 'lv_pixels': [], 'my_pixels': []})
for i in tqdm(range(int(fullImgList.shape[0]/10000)+1)):
    imgInBatch = fullImgList[(10000*i):(10000*(i+1))]
    trainedModel = load_learner(name, 'model.pkl')
    trainedModel.data.add_test(SegmentationItemList.from_df(imgInBatch,path="nifti",cols="file"), tfm_y=False)
    predictions,_=trainedModel.get_preds(DatasetType.Test)
    predictions = predictions.argmax(dim=1)
    lv_pixels = (predictions==1).sum(dim=(1,2))
    my_pixels = (predictions==2).sum(dim=(1,2))
    pixelTable = pd.concat([pixelTable, pd.DataFrame({'file': trainedModel.data.test_ds.items, 'lv_pixels': lv_pixels, 'my_pixels': my_pixels})])

pixelTable.to_csv(name+"/predictions.tsv",sep="\t",index=False)

