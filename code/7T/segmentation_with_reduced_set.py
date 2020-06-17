import argparse
from fastai.vision import *
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
from fastai.callbacks import CSVLogger

parser = argparse.ArgumentParser()
parser.add_argument('--set', dest='set', help='subset to work on (v7, v3, v1, r7, r3, r1, esed, r_esed)', type=str)
parser.add_argument('--cuda_device', dest='cuda_device', help='cuda device index', default=0, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.cuda_device)
subsets = pd.read_csv("image_subsets.tsv", sep="\t", index_col=0)
#tfms = get_transforms(do_flip=False,max_rotate=20,max_lighting=.4,max_zoom=1.2)
# use extreme transforms
tfms = get_transforms(do_flip=True,max_rotate=90,max_lighting=.4,max_zoom=1.2)

get_y_fn = lambda x: str(x).replace("image", "mask")

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

def standard_training(model, name, epochs, freeze):
    frozen = "unfrozen"
    if(freeze):
        frozen = "frozen"
        model.freeze_to(-1)
    
    lr_find(model)
    model.recorder.plot()

    lr=1e-4
    model.fit_one_cycle(epochs, lr)
    model.save('{}-{}-{}'.format(name,frozen,epochs))

    model.unfreeze()
    
    lr_find(model)
    model.recorder.plot()

    unfrozen_epochs = epochs
    if not freeze:
        unfrozen_epochs = 2*epochs
    lr=1e-5
    model.fit_one_cycle(epochs, lr)
    model.save('{}-unfrozen-{}'.format(name,unfrozen_epochs))

    model.export(name+'.pkl',destroy=True) 
    
def learn_with_reduced_set(redset, subsets, correct_epochs=True):
    name = "doubleTransferLearn_{}".format(redset)
    epochs = 10
    if correct_epochs:
        epochs = int(round(10 * len(subsets[redset]) / subsets[redset].sum()))
    
    src_red = (SegmentationItemList.from_folder("images", presort=True)
           .split_by_folder(train="train", valid="val")
           .label_from_func(get_y_fn, classes=np.array(["background","left_ventricle","myocardium"])))

    print(src_red)
    src_red.train.filter_by_func(lambda x,y: not subsets[redset][x.name])
    print(src_red)
    
    data_red = (src_red.transform(tfms,size=256,padding_mode="zeros",resize_method=ResizeMethod.PAD,tfm_y=True)
           .add_test_folder("test", tfm_y=False)
           .databunch(bs=8)
           .normalize(imagenet_stats))

    baseModel = load_learner("resnet34_5percent_size256_extremeTfms_ceLoss", 'model.pkl')
    baseModel.data = data_red
    baseModel.path = Path('images')
    baseModel.callback_fns[1]=partial(CSVLogger, append=True, filename=name+"_log")
    standard_training(baseModel, name, epochs, freeze=True)
    
    name = "imagenetTransferLearn_{}".format(redset)
    baseModel = unet_learner(data_red, models.resnet34, pretrained=True, metrics=[acc_seg,dice0inv,dice1,dice2], callback_fns=[partial(CSVLogger, append=True, filename=name+"_log")])
    standard_training(baseModel, name, epochs, freeze=True)
    
    name = "plainLearn_{}".format(redset)
    baseModel = unet_learner(data_red, models.resnet34, pretrained=False, metrics=[acc_seg,dice0inv,dice1,dice2], callback_fns=[partial(CSVLogger, append=True, filename=name+"_log")])
    standard_training(baseModel, name, epochs, freeze=False)

learn_with_reduced_set(args.set, subsets, correct_epochs=True)