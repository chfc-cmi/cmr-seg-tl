dependencies = ['torch','fastai']
import torch
import tempfile
from fastai.vision import load_learner

# cmr_seg_base is the name of entrypoint
def cmr_seg_base(**kwargs):
    """ # This docstring shows up in hub.help()
    Base model pretrained on the Second Annual Data Science Bowl
    Cardiac Challenge Data: https://www.kaggle.com/c/second-annual-data-science-bowl
    labeled by ukbb_cardiac network https://github.com/baiwenjia/ukbb_cardiac
    """
    # Call the model, load pretrained weights
    url = "https://github.com/chfc-cmi/cmr-seg-tl/releases/download/v0.1.0/resnet34_5percent_size256_extremeTfms_ceLoss.pkl"
    dst = tempfile.NamedTemporaryFile()
    torch.hub.download_url_to_file(url,dst.name,progress=True)
    model = load_learner(path=tempfile.gettempdir(),file=dst.name,tfm_y=False)
    return model
