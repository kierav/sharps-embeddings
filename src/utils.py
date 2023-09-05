""" Utils functions """

import os
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import random


def split_data(df,val_split,test=''):
    """
        Split dataset into training, validation, hold-out (pseudotest) and test sets.
        The test set can be either 'test_a' which is all data from November and December,
        or 'test_b' which is 2021-2023, or both combined. The hold-out set is data from
        Sept 15-Oct 31. The remaining data is split temporally 4:1 into training and 
        validation.

        Parameters:
            df (dataframe):     Pandas dataframe containing all the data
            val_split (0-4):    Number between 0-4 indicating which temporal training/validation split to select    
            test (str):         Which test set to choose ('test_a' or 'test_b', otherwise both)

        Returns:
            df_test (dataframe):        Test set
            df_pseudotest (dataframe):  Hold-out set
            df_train (dataframe):       Training set
            df_val (dataframe):         Validation set
    """

    # hold out test sets
    inds_test_a = (df['sample_time'].dt.month >= 11)
    inds_test_b = (df['sample_time']>=datetime(2021,1,1))
    
    # select test set
    if test == 'test_a':
        inds_test = inds_test_a
    elif test == 'test_b':
        inds_test = inds_test_b
    else:
        inds_test = inds_test_a | inds_test_b

    df_test = df.loc[inds_test,:]
    df_full = df.loc[~inds_test,:]

    # select pseudotest/hold-out set
    if test == 'test_a':
        inds_pseudotest = ((df_full['sample_time'].dt.month==10)&(df_full['sample_time'].dt.day>26)) | ((df_full['sample_time'].dt.month==1)&(df_full['sample_time'].dt.day<6))
    elif test == 'test_b':
        inds_pseudotest = (df['sample_time']>=datetime(2020,12,26))
    else:
        inds_pseudotest = (df_full['sample_time'].dt.month==10) | ((df_full['sample_time'].dt.month==9)&(df_full['sample_time'].dt.day>15))
    # inds_pseudotest = (df['sample_time']<datetime(1996,1,1)) | ((df_full['sample_time'].dt.month==10)&(df_full['sample_time'].dt.day>26)) | ((df_full['sample_time'].dt.month==1)&(df_full['sample_time'].dt.day<6))
    df_pseudotest = df_full.loc[inds_pseudotest,:]

    # split training and validation
    df_train = df_full.loc[~inds_pseudotest,:]
    df_train = df_train.reset_index(drop=True)
    n_val = int(np.floor(len(df_train)/5))
    df_val = df_train.iloc[val_split*n_val:(val_split+1)*n_val,:]
    df_train = df_train.drop(df_val.index)

    return df_test,df_pseudotest,df_train,df_val

def diverse_sampler(self, filenames, features, n):
    """
    Parameters:
        filenames(list): filename (SHARPs)
        features (list): embedded data/SHARP parameters
        n (int): number of points to sample from the embedding space

    Returns:

        result (list): list of n points sampled from the embedding space

    Ref:
        https://arxiv.org/pdf/2107.03227.pdf

    """
    filenames_ = filenames.copy()
    features_ = features.copy()
    result = [random.choice(features_)]
    filenames_results = [random.choice(filenames_)]
    distances = [1000000] * len(features_)
    
    for _ in range(n):
        dist = np.sum((features_ - result[-1])**2, axis=1)**0.5
        for i in range(features_.shape[0]):
            if distances[i] > dist[i]:
                distances[i] = dist[i]
        idx = distances.index(max(distances))
        result.append(features_[idx])
        filenames_results.append(filenames_[idx])
        
        features_ = np.delete(features_, idx, axis=0)
        del filenames_[idx]
        del distances[idx]

    return filenames_results[1:], np.array(result[1:])


def save_predictions(preds,dir,appendstr:str=''):
    """
    Save predicted files and embeddings
    
    Parameters:
        preds:  output of model predict step (as list of batch predictions)
        dir:    directory for saving
        appendstr: string to save at end of filename
    Returns:
        embeddings (np array):      output of model embed step 
    """
    file = []
    embeddings = []
    for predbatch in preds:
        file.extend(predbatch[0])
        embeddings.extend(np.array(predbatch[1]))
    embeddings = np.array(embeddings)

    df = pd.DataFrame({'embed'+str(i):embeddings[:,i] for i in range(np.shape(embeddings)[1])})
    df.insert(0,'filename',file)
    df.to_csv(dir+os.sep+'embeddings'+appendstr+'.csv',index=False)

    return file, embeddings


def load_model(ckpt_path,modelclass,api):
    """
    Load model into wandb run by downloading and initializing weights

    Parameters:
        ckpt_path:  wandb path to download model checkpoint from
        model:      model class
        api:        instance of wandb Api
    Returns:
        model:      Instantiated model class object with loaded weights
    """
    print('Loading model checkpoint from ', ckpt_path)
    artifact = api.artifact(ckpt_path,type='model')
    artifact_dir = artifact.download()
    model = modelclass.load_from_checkpoint(artifact_dir+'/model.ckpt',map_location='cpu')
    return model