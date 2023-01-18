# Make predictions with the pre-trained base model

In order to make predictions with the base model on your own data, you can either load the model through torch hub:

```python
import torch
trainedModel = torch.hub.load("chfc-cmi", "cmr_seg_base")
```

or download the model file from the [releases page](https://github.com/chfc-cmi/cmr-seg-tl/releases) and load it with fastai:

```python
from fastai.vision.all import load_learner
trainedModel = load_learner("resnet34_5percent_size256_extremeTfms_ceLoss_fastai2.pkl")
```

The `trainedModel` can directly be used to make predictions on an image file (or images loaded in python):

```python
pred = trainedModel.predict("path/to/image.png")
```

:warning: The input image is resized (with zero padding) to 256x256 before prediction, thus the resulting mask is 256x256, as well. For more details about the `predict` function and it's return values, see the [fastai documentation](https://docs.fast.ai/learner.html#learner.predict).
