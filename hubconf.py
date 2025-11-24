dependencies = ['torch','fastai']
import torch
import os
import tempfile
from fastai.vision.learner import load_learner

# cmr_seg_base is the name of entrypoint
def cmr_seg_base(**kwargs):
    """ # This docstring shows up in hub.help()
    Base model pretrained on the Second Annual Data Science Bowl
    Cardiac Challenge Data: https://www.kaggle.com/c/second-annual-data-science-bowl
    labeled by ukbb_cardiac network https://github.com/baiwenjia/ukbb_cardiac
    """
    # Call the model, load pretrained weights
    url = "https://github.com/chfc-cmi/cmr-seg-tl/releases/download/v0.5.0/resnet34_5percent_size256_extremeTfms_ceLoss_fastai2.pkl"
    dst_dir = torch.hub.get_dir() + '/chfc-cmi_cmr-seg-tl_master/'
    os.makedirs(dst_dir, exist_ok=True)
    dst = dst_dir + 'resnet34_5percent_size256_extremeTfms_ceLoss_fastai2.pkl'
    if not os.path.isfile(dst):
        torch.hub.download_url_to_file(url,dst,progress=True)
    model = load_learner(fname=dst)
    return model
