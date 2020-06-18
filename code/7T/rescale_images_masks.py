from fastai.vision import *
from tqdm.notebook import tqdm
from pathlib import Path

data = (SegmentationItemList.from_folder("images", presort=True)
       .split_none()
       .label_from_func(lambda x: str(x).replace("image", "mask"), classes=np.array(["background","left_ventricle","myocardium"]))
       .transform(None,size=256,padding_mode="zeros",resize_method=ResizeMethod.PAD,tfm_y=True)
       .databunch(bs=8)
       .normalize(imagenet_stats))

for i in tqdm(range(len(data.train_ds))):
    fname = str(data.train_ds.items[i])
    scaled_img = data.train_ds[i][0]
    scaled_mask = data.train_ds[i][1]
    scaled_img.save("scaled_"+fname)
    scaled_mask.save("scaled_"+fname.replace("image","mask"))

# Repeat for additional unscaled masks e.g. from AutoQ or second expert