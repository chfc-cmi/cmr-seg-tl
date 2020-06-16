import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

volunteers = pd.read_csv("resolution.tsv",sep="\t",index_col="id")

os.makedirs('../masks', exist_ok=True)
Image.new('I', (472,512), 0).save('../masks/empty_472x512.png')
Image.new('I', (512,472), 0).save('../masks/empty_512x472.png')

for volunteerId,volunteer in tqdm(volunteers.iterrows()):
    contour = pd.read_csv("{}.tsv".format(volunteerId),sep=" ",names=["x","y","z","t","c"],usecols=range(5))
    iters = contour.iloc[:,2:4].drop_duplicates().to_numpy()
    for i in tqdm(iters, leave=False):
        z = i[0]
        t = i[1]
        poly = [(x[0],x[1]) for x in contour[contour.z==z][contour.t==t][contour.c==1].to_numpy()[:,0:2]]
        img = Image.new('L', (volunteer["columns"], volunteer["rows"]), 0)
        if(len(poly)>1):
            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        mask = np.array(img)
        poly2 = [(x[0],x[1]) for x in contour[contour.z==z][contour.t==t][contour.c==0].to_numpy()[:,0:2]]
        img = Image.new('L', (volunteer["columns"], volunteer["rows"]), 0)
        if(len(poly2)>1):
            ImageDraw.Draw(img).polygon(poly2, outline=1, fill=1)
        mask2 = np.array(img)
        im_array = 2*mask.astype(np.int32)-mask2
        im = Image.fromarray(im_array, 'I')
        im.save('../masks/{}_slice{:03d}_frame{:03d}-mask.png'.format(volunteerId,z,t))
