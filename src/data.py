from datetime import datetime,timedelta
import glob
import torch
import torchvision.transforms.v2 as transforms
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader
from utils import split_data

class SharpsDataset(Dataset):
    """
        Pytorch dataset for handling magnetogram tile data 
        
    """
    def __init__(self, df: pd.DataFrame, transform, feature_cols:list=None, maxval:int=1000):
        '''
            Initializes image files in the dataset
            
            Args:
                df (dataframe):     Pandas dataframe containing filenames and labels 
                transform:          single or list of torchvision transforms
                feature_cols (list): names of scalar feature columns
        '''
        self.name_frame = df.loc[:,'file']
        if feature_cols == None:
            feature_cols = []
        self.features = df.loc[:,feature_cols]
        self.transform = transform
        self.maxval = maxval

    def __len__(self):
        '''
            Calculates the number of images in the dataset
                
            Returns:
                int: number of images in the dataset
        '''
        return len(self.name_frame)

    def __getitem__(self, idx):
        '''
            Retrieves an image from the dataset and creates a copy of it,
            applying a series of random augmentations to the copy.

            Args:
                idx (int): index of the image to retrieve
                
            Returns:
                tuple: (image, image2) where image2 is an augmented modification
                of the input and image can be the original image, or another augmented
                modification, in which case image2 is double augmented
        '''
        file = self.name_frame.iloc[idx]
        image = np.array(h5py.File(file,'r')['hmi']).astype(np.float32)
        image = np.nan_to_num(image)

        # Clip and normalize magnetogram data
        image = (np.clip(image,-self.maxval,self.maxval)/self.maxval+1)/2

        image = np.transpose(image,(1,2,0))
        image = self.transform(image)    

        features = torch.Tensor(self.features.iloc[idx])

        return file,image,features

    
class SharpsDataModule(pl.LightningDataModule):
    """
    Datamodule for self supervision on tiles dataset
    """

    def __init__(self,data_file:str,batch:int=128,
                dim:int=128,val_split:int=0,test:str='',
                features:list=None,maxval:int=1000):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch
        self.val_split = val_split
        self.test = test
        if features == None:
            features = []
        self.features = features
        self.maxval = maxval
        self.feat_scaler = StandardScaler()

        # define data transforms - augmentation for training
        self.training_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485,),std=(0.229,)),
            transforms.RandomInvert(p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([
                 transforms.RandomRotation(degrees=45,fill=0.5)
                 ]),p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.Resize((dim,dim),antialias=True),
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485,),std=(0.229,)),
            transforms.Resize((dim,dim),antialias=True),
        ])

    def prepare_data(self):
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'],format='mixed')

        
    def setup(self,stage:str):
        # split into training and validation the same as for forecasting
        df_test,df_pseudotest,self.df_train,df_val = split_data(self.df,self.val_split,self.test)

        # use training+val together and pseudotest as validation
        self.df_train = pd.concat([self.df_train,df_val])

        # scale features
        self.df_train[self.features] = self.feat_scaler.fit_transform(self.df_train[self.features])
        df_pseudotest[self.features] = self.feat_scaler.transform(df_pseudotest[self.features])
        df_test[self.features] = self.feat_scaler.fit_transform(df_test[self.features])

        self.train_set = SharpsDataset(self.df_train,self.training_transform,self.features,maxval=self.maxval)
        self.val_set = SharpsDataset(df_pseudotest,self.transform,maxval=self.maxval)
        self.test_set = SharpsDataset(df_test,self.transform,self.features,maxval=self.maxval)
        print('Train:',len(self.train_set),
              'Valid:',len(self.val_set),
              'Test:',len(self.test_set))

        
    def subsample_trainset(self,filenames):
        # given a list of filenames, subsample so the train set only includes files from that list
        subset_df = self.df_train[self.df_train['file'].isin(filenames)]
        self.subset_train_set = SharpsDataset(subset_df,self.training_transform,self.features,maxval=self.maxval)

    def subset_train_dataloader(self,shuffle=True):
        return DataLoader(self.subset_train_set,batch_size=self.batch_size,num_workers=4,shuffle=shuffle,drop_last=True)
    
    def train_dataloader(self,shuffle=True):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,shuffle=shuffle,drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,num_workers=4,drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)
