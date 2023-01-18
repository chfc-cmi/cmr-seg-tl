# Re-train the pre-trained base model on own data

In order to get the base model, you can either load the model through torch hub:

```python
import torch
trainedModel = torch.hub.load("chfc-cmi", "cmr_seg_base")
```

or download the model file from the [releases page](https://github.com/chfc-cmi/cmr-seg-tl/releases) and load it with fastai:

```python
from fastai.vision.all import load_learner
trainedModel = load_learner("resnet34_5percent_size256_extremeTfms_ceLoss_fastai2.pkl")
```

Next, you have to define your own `dataloaders`. This can be done through the `DataBlock` API of `fastai`. Here is an example, assuming, that you have the data in a structure like this:

```
data
├── images
│   ├── train
│   │   ├── img-1.png
│   │   └── …
│   └── val
│       ├── img-1.png
│       └── …
└── masks
    ├── train
    │   ├── img-1.png
    │   └── …
    └── val
        ├── img-1.png
        └── …

```

We can get appropriate data loaders with this code:

```python
def label_func(x):
    return str(x).replace("image","mask")

def get_parent(x):
    return Path(x).parent.name == 'val'

dbl = DataBlock(blocks=(ImageBlock, MaskBlock(codes = np.array(["background","left_ventricle","myocardium"]))),
        get_items = get_image_files,
        get_y = label_func,
        splitter = FuncSplitter(get_parent),
        item_tfms=Resize(512, method='crop'),
        batch_tfms=aug_transforms(do_flip=True,max_rotate=90,max_lighting=.4,max_zoom=1.2,size=256))

dls = dbl.dataloaders("data/images", bs=16)
```

Now, you can replace the existing dataloaders with the new ones on the `trainedModel`:

```python
trainedModel.dls = dls
```

Now, the trained model can be used to make predictions on the whole validation set (without re-training), and to re-train it:

```python
# look at results, without re-training
trainedModel.show_results()
val_preds = trainedModel.get_preds()

# continue training
trainedModel.freeze()
trainedModel.lr_find()
trainedModel.fit_one_cycle(10, lr_max=1e-4)
traindeModel.unfreeze()
trainedModel.lr_find()
trainedModel.fit_one_cycle(10, lr_max=1e-5)
```
