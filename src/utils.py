""" Utils functions """

import os
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,confusion_matrix,precision_recall_curve,r2_score
from sklearn.calibration import calibration_curve

# color-blind friendly palette https://davidmathlogic.com/colorblind/#%23004A98-%23C5C5C5-%23FF538C-%234DC6FF-%23FFA42C-%23C050FF-%2375FF84-%23C70000-%233D98C3-%235D5D5D
Clr = ['#004A98',
       '#C5C5C5',
       '#FF538C',
       '#4DC6FF',
       '#FFA42C',
       '#C050FF',
       '#75FF84',
       '#C70000'
       ]

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

def diverse_sampler(filenames, features, n):
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
    result = np.tile(random.choice(features_),(n+1,1))
    filenames_results = [None]*(n+1)
    distances = [1000000] * len(features_)

    for i in range(n):
        dist = np.sum((features_ - np.mean(result[-2:],axis=0))**2, axis=1)**0.5
        distances = np.minimum(distances,dist)
        idx = np.argmax(distances)
        result[i+1,:] = features_[idx,:]
        filenames_results[i+1] = filenames_[idx]
        
        features_ = np.delete(features_, idx, axis=0)
        filenames_ = np.delete(filenames_, idx, axis=0)
        distances = np.delete(distances, idx, axis=0)

    return filenames_results[1:], np.array(result[1:])


def save_predictions(preds,dir,appendstr:str=''):
    """
    Save predicted files and embeddings
    
    Parameters:
        preds:  output of model predict step (as list of batch predictions)
        dir:    directory for saving
        appendstr: string to save at end of filename
    Returns:
        file (list):            list of filenames
        embeddings (list):      output of model embed step 
        df:                     dataframe including files and embeddings
    """
    file = []
    embeddings = []
    for predbatch in preds:
        file.extend(predbatch[0])
        embeddings.extend(np.array(predbatch[1]))
    embeddings = np.array(embeddings)

    df = pd.DataFrame({'embed'+str(i):embeddings[:,i] for i in range(np.shape(embeddings)[1])})
    df.insert(0,'file',file)
    df.to_csv(dir+os.sep+'embeddings'+appendstr+'.csv',index=False)

    return file, embeddings,df


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

def reliability_diag(ytrue,ypred,ax,label,nbins=10,plot=True,plot_hist=False,**kwargs):
    """
    Plots a reliability diagram (calibration curve) on a given axis and computes
    the calibration error metrics (expected calibration error and max calibration error)

    Parameters:
        ytrue (np array):       true values
        ypred (np array):       predicted values
        ax (axis object):       matplotlib axis for plotting
        label (str):            label for plot object
        nbins (int):            number of bins to compute the reliability diagram on
        plot (bool):            whether or not to plot
        plot_hist (bool):       whether or not to overlay a histogram of samples 
        **kwargs:               additional arguments for matplotlib plot function

    Returns:
        ece (float):            expected calibration error
        mce (float):            max calibration error
    """
    
    prob_true, prob_pred = calibration_curve(ytrue,ypred,n_bins=nbins)
    mce = max(abs(prob_true-prob_pred))

    if plot:
        ax.plot(prob_pred,prob_true,'.-',label=label,**kwargs)

    bin_edges = prob_pred[:-1] + np.diff(prob_pred)/2
    bin_edges = np.insert(bin_edges,0,0)
    bin_edges = np.append(bin_edges,1)
    ni = np.histogram(ypred,bins=bin_edges)[0]

    if plot_hist:
        sns.histplot(ypred,ax=ax,bins=bin_edges,alpha=0.6,stat='probability',label='_',**kwargs)

    ece = sum(ni*abs(prob_true-prob_pred))/sum(ni)

    return ece,mce

def plot_performance(df,cal='yprob',nbins=5):
    """
    Plots a panel of figures illustrating model performance for an ensemble,
    reliability diagram, TPR vs FPR and precision vs. recall
    
    Parameters:
        df (dataframe):     dataframe assembled from create_ensemble_df routine
        cal (str):          label of the calibrated probabilities
        nbins (int):        number of bins for the reliability diagram
    """
    fpr,tpr,thresh = roc_curve(df['ytrue'],df[cal+'_median'])
    pr,re,thresh2 = precision_recall_curve(df['ytrue'],df[cal+'_median'])

    tss = []
    hss = [] 
    for t in np.arange(0,1,0.02):
        tn, fp, fn, tp = confusion_matrix(df['ytrue'], df[cal+'_median']>=t).ravel()
        tss.append((tp) / (tp + fn) - (fp) / (fp + tn))
        hss.append(2*(tp*tn-fp*fn)/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn)))
    
    fig,ax = plt.subplots(1,3,figsize=(9,3))

    ax[0].plot([0,1],[0,1],'-k',linewidth=1,label='_')
    ax[0].plot([0,1],[sum(df['ytrue']/len(df)),sum(df['ytrue']/len(df))],'--k',linewidth=1,label='_')
    ax[0].plot([0,1],[sum(df['ytrue']/len(df))/2,sum(df['ytrue']/len(df))/2+0.5],'--k',linewidth=1,label='_')
    for model in range(5):
        reliability_diag(df['ytrue'],df[cal+str(model)],ax[0],label='_',nbins=nbins,color=Clr[model])
    reliability_diag(df['ytrue'],df[cal+'_median'],ax[0],label='Median',nbins=nbins,color=Clr[5],marker='*',markersize=8)
    sns.histplot(df[cal+'_median'],ax=ax[0],stat='probability',binwidth=0.1,binrange=(0,1),color=Clr[5],alpha=0.5)
    ax[0].set_xlabel('Probability')
    ax[0].set_ylabel('Flare frequency')
    ax[0].legend()
    ax[0].set_ylim([-0.05,1.05])
    ax[0].set_xlim([-0.05,1.05])

    ax[1].plot(fpr,fpr,'--k',linewidth=1)
    ax[1].plot(fpr,tpr,'-')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_ylim([-0.05,1.05])
    ax[1].set_xlim([-0.05,1.05])

    ax[2].plot([0,1],[sum(df['ytrue']/len(df)),sum(df['ytrue']/len(df))],'--k',linewidth=1,label='_')
    ax[2].plot(re,pr)
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_ylim([-0.05,1.05])
    ax[2].set_xlim([-0.05,1.05])

    plt.tight_layout()

def print_metrics(ypred,y,thresh:float=0.5,print_results:bool=True):
    """
    Print metrics for probabilistic forecasting

    Parameters:
        ypred       predicted probabilities
        y           true outcomes (0 or 1)
        thresh      threshold value for calculating binary metrics (between 0 and 1)
    
    Returns:
        metrics     list of metrics (MSE,BSS,APS,Gini,TSS,HSS,TPR,FPR)
    """

    # probabilistic metrics
    mse = (sum((ypred-y)**2))/len(ypred)
    bss = (sum((ypred-y)**2)-sum((sum(y)/len(y)-y)**2))/(-sum((sum(y)/len(y)-y)**2))
    aps = average_precision_score(y,ypred)
    gini = 2*roc_auc_score(y,ypred)-1

    # binary metrics
    ypred_binary = ypred>=thresh
    C = confusion_matrix(y,ypred_binary)
    tp = C[1,1]
    tn = C[0,0]
    fp = C[0,1]
    fn = C[1,0]
    tss = (tp) / (tp + fn) - (fp) / (fp + tn)
    hss = 2*(tp*tn-fp*fn)/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    if print_results:
        print(f'MSE:{mse:0.3f}, BSS:{bss:0.3f}, APS:{aps:0.3f}, Gini:{gini:0.3f}',
          f'TSS:{tss:0.3f}, HSS:{hss:0.3f}, TPR:{tpr:0.3f}, FPR:{fpr:0.3f}')
    
    return [mse,bss,aps,gini,tss,hss,tpr,fpr]