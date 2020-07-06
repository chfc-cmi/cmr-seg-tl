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

    # Call the model, load pretrained weights
    url = "https://github.com/chfc-cmi/cmr-seg-tl/releases/download/v0.1.0/resnet34_5percent_size256_extremeTfms_ceLoss.pkl"
    dst = tempfile.NamedTemporaryFile()
    torch.hub.download_url_to_file(url,dst.name,progress=True)
    model = load_learner(path=tempfile.gettempdir(),file=dst.name,tfm_y=False)
    return model
