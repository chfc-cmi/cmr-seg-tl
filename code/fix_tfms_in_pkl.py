from fastai.vision.all import load_learner


def label_func(x):
    pass


def acc_seg(input, target):
    pass


def diceComb(input, targs):
    pass


def diceLV(input, targs):
    pass


def diceMY(input, targs):
    pass


def model_file(version):
    return f"/tmp/resnet34_5percent_size256_extremeTfms_ceLoss_fastai2_{version}.pkl"


trainedModel_020 = load_learner(model_file("0.2.0"))
trainedModel_040 = load_learner(model_file("0.4.0"))

trainedModel_040.dls[0].after_batch = trainedModel_020.dls[0].after_batch
trainedModel_040.dls[1].after_batch = trainedModel_020.dls[1].after_batch

trainedModel_040.export(model_file("0.5.0"))
