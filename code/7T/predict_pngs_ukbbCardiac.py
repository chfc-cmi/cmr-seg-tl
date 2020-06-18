# Prediction via UKBB_cardiac
# Needs the ukbb_cardiac repo in the current folder (incl. pretrained model)
# Paths and parameters are hard coded in lower part
import os
import math
import numpy as np
import tensorflow.compat.v1 as tf
from glob import glob
from tqdm import tqdm
import PIL
import imageio
from pathlib import Path
from ukbb_cardiac.common.image_utils import rescale_intensity

def img_from_png(file):
    """ Reads a png file and returns a numpy array suitable for prediction via ukbb_cardiac
    it also returns information required for rescaling the prediction mask to match the image file dimensions
    """
    img_from_png = imageio.imread(file)
    # flip and transpose (png directly from dicom differs from directions in nifti as used by ukbb_cardiac)
    img_from_png = np.flip(img_from_png.transpose(1,0,2),axis=1)[:,:,0]
    # intensity scaling
    img_from_png = rescale_intensity(img_from_png,(1,99))
    # padding to multiple of 16
    X, Y = img_from_png.shape
    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
    img_from_png = np.pad(img_from_png, ((x_pre, x_post), (y_pre, y_post)), 'constant')
    # fix dimensions for single channel and single image in batch
    img_from_png = np.expand_dims(img_from_png,0)
    img_from_png = np.expand_dims(img_from_png,-1)
    return img_from_png, (x_pre,X,y_pre,Y)

def prediction_to_png(pred, pad_region, filename, outdir):
    (x_pre,X,y_pre,Y) = pad_region
    pred_out = pred[0,x_pre:x_pre + X, y_pre:y_pre + Y]
    pred_out = np.flip(pred_out,1).transpose(1,0)
    PIL.Image.fromarray(pred_out.astype(np.int32),'I').save('{}/bg_lv_my_rv/{}'.format(outdir,filename))
    pred_out[pred_out==3] = 0
    PIL.Image.fromarray(pred_out.astype(np.int32),'I').save('{}/bg_lv_my/{}'.format(outdir,filename))

pngs = glob("images_scaled/*/*.png")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('ukbb_cardiac/trained_model/FCN_sa.meta')
    saver.restore(sess, 'ukbb_cardiac/trained_model/FCN_sa')
    for png in tqdm(pngs):
        tmp_img, tmp_pad = img_from_png(png)
        _, tmp_pred = sess.run(['prob:0', 'pred:0'],
                       feed_dict={'image:0': tmp_img, 'training:0': False})
        prediction_to_png(tmp_pred, tmp_pad, Path(png).name.replace('image','mask'), "ukbb_preds")