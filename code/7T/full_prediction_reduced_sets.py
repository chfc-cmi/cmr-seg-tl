from fastai.vision import *
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from fastai.callbacks import CSVLogger

torch.cuda.set_device(1)

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

get_y_fn = lambda x: str(x).replace("image", "mask")

# All images as testData for full prediction
testData = ImageList.from_folder("images")
testData

def make_full_predictions(modelFile, name):
    learn = load_learner('images',modelFile,test=testData,tfm_y=False)
    preds,_ = learn.get_preds(DatasetType.Test)
    for d in ['preds{}/{}/{}'.format(i,name,j) for i in ['', '_raw'] for j in ['test','val','train']]:
        os.makedirs(d,exist_ok=True)
    for i in tqdm(range(len(preds))):
        p = preds[i]
        file = str(testData.items[i])
        raw_file = file.replace("images","preds_raw/{}".format(name))
        Image(p).save(raw_file)
        pred_file = file.replace("images","preds/{}".format(name))
        PIL.Image.fromarray(np.array(p.argmax(dim=0), dtype=np.int32),'I').save(pred_file)
    return preds

tmp = make_full_predictions('doubleTransferLearn_esed.pkl', 'doubleTransferLearn_esed')
tmp = make_full_predictions('doubleTransferLearn_r1.pkl', 'doubleTransferLearn_r1')
tmp = make_full_predictions('doubleTransferLearn_r3.pkl', 'doubleTransferLearn_r3')
tmp = make_full_predictions('doubleTransferLearn_r7.pkl', 'doubleTransferLearn_r7')
tmp = make_full_predictions('doubleTransferLearn_r_esed.pkl', 'doubleTransferLearn_r_esed')
tmp = make_full_predictions('doubleTransferLearn_v1.pkl', 'doubleTransferLearn_v1')
tmp = make_full_predictions('doubleTransferLearn_v3.pkl', 'doubleTransferLearn_v3')
tmp = make_full_predictions('doubleTransferLearn_v7.pkl', 'doubleTransferLearn_v7')
tmp = make_full_predictions('imagenetTransferLearn_esed.pkl', 'imagenetTransferLearn_esed')
tmp = make_full_predictions('imagenetTransferLearn_r1.pkl', 'imagenetTransferLearn_r1')
tmp = make_full_predictions('imagenetTransferLearn_r3.pkl', 'imagenetTransferLearn_r3')
tmp = make_full_predictions('imagenetTransferLearn_r7.pkl', 'imagenetTransferLearn_r7')
tmp = make_full_predictions('imagenetTransferLearn_r_esed.pkl', 'imagenetTransferLearn_r_esed')
tmp = make_full_predictions('imagenetTransferLearn_v1.pkl', 'imagenetTransferLearn_v1')
tmp = make_full_predictions('imagenetTransferLearn_v3.pkl', 'imagenetTransferLearn_v3')
tmp = make_full_predictions('imagenetTransferLearn_v7.pkl', 'imagenetTransferLearn_v7')
tmp = make_full_predictions('plainLearn_esed.pkl', 'plainLearn_esed')
tmp = make_full_predictions('plainLearn_r1.pkl', 'plainLearn_r1')
tmp = make_full_predictions('plainLearn_r3.pkl', 'plainLearn_r3')
tmp = make_full_predictions('plainLearn_r7.pkl', 'plainLearn_r7')
tmp = make_full_predictions('plainLearn_r_esed.pkl', 'plainLearn_r_esed')
tmp = make_full_predictions('plainLearn_v1.pkl', 'plainLearn_v1')
tmp = make_full_predictions('plainLearn_v3.pkl', 'plainLearn_v3')
tmp = make_full_predictions('plainLearn_v7.pkl', 'plainLearn_v7')
