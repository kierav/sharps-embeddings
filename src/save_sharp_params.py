import pytorch_lightning as pl
from dataset import SHARPdataset
import numpy as np
import pandas as pd

IMAGE_SIZE = 128

DATA_PATH = '/d0/kvandersande/sharps_hdf5/'
dataset = SHARPdataset(DATA_PATH, image_size=IMAGE_SIZE)

df = {'filename':[], 'pflux':[], 'nflux':[], 'parea':[], 'narea':[]}

L = len(dataset)

for i in range(L):
    (image, filename) = dataset[i]
    image = image[0,:,:]
    pf = ((image-0.5)*(image>0.5)).sum().item()
    pa = (image>0.5).sum().item()
    nf = ((0.5-image)*(image<0.5)).sum().item()
    na = (image<0.5).sum().item()
    df['filename'].append(filename)
    df['pflux'].append(pf)
    df['parea'].append(pa)
    df['nflux'].append(nf)
    df['narea'].append(na)
    k = i+1
    print(filename, pf, pa, nf, na)
    print(f'{k} done out of {L}')
    
print(df)
df = pd.DataFrame(df)
df.to_csv(f'/d0/subhamoy/models/vae/sharps/sharp_params_resize_{IMAGE_SIZE}.csv')