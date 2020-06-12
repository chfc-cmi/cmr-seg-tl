from tqdm import tqdm
import imageio
from PIL import Image
from glob import glob
import numpy as np

for file in tqdm(glob("masks/*/*.png")):
    target = file.replace("masks/","masks_2class/")
    tmp_im = imageio.imread(file)
    tmp_norm = np.mod(tmp_im,3).astype(np.int32)
    im = Image.fromarray(tmp_norm, 'I')
    im.save(target)
