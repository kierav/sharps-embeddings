import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import h5py
import torch, torchvision

class SHARPdataset(Dataset):

    def __init__(self, data_path: str, param: bin=False,
                 data_stride:int = 1, image_size:int = 256,
                 datatype=np.float32):
        '''
            Initializes image files in the dataset
            
            Args:
                data_path (str): path to the folder containing the SHARP h5
                data_stride (int): stride to use when loading the images to work 
                                    with a reduced version of the data
                datatype (numpy.dtype): datatype to use for the images
                param (bin): add SHARP parameters
        '''
        self.data_path = data_path
        self.sharp_images = glob.glob(data_path + "/*.h5")
        if data_stride>1:
            self.sharp_images = self.sharp_images[::data_stride]
        self.image_size = image_size
        self.param = param
        self.datatype = datatype
        if self.param is True:
            'read the csv of SHARP params'

    def __len__(self):
        '''
            Calculates the number of images in the dataset
                
            Returns:
                int: number of images in the dataset
        '''
        return len(self.sharp_images)

    def __getitem__(self, idx):
        '''
            Retrieves an image from the dataset and creates a copy of it,
            applying a series of random augmentations to the copy.

            Args:
                idx (int): index of the image to retrieve
                
            Returns:
            
        '''
        file = h5py.File(self.sharp_images[idx])
        key = list(file.keys())[0]
        data = np.array(file[key])
        data = (np.clip(data, -5000, 5000)/5000 + 1)/2
        data = data[None, :, :, :]
        data = torch.from_numpy(data.astype(self.datatype))
        resize = torchvision.transforms.Resize((self.image_size, self.image_size), antialias=True)
        data = resize(data)
        
        if self.param is True:
            return ('SHARP_PARAMS [idx]', data)
        else:
            return data, self.sharp_images[idx]