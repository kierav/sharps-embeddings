import numpy as np
import os
import torchvision
from torch import nn
import wandb
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.preprocessing import MinMaxScaler, normalize
from data import SharpsDataModule
from autoencoder import SharpEmbedder,Encoder,Decoder
from lr_model import LinearModel
from utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import random
import torch
import yaml

torchvision.disable_beta_transforms_warning()

def main():    
    # read in config file
    with open('experiment_config.yml') as config_file:
        config = yaml.safe_load(config_file.read())
    

    run = wandb.init(config=config,project=config['meta']['project'],
                        name=config['meta']['name'],
                        group=config['meta']['group'],
                        tags=config['meta']['tags'],
                        )
    config = wandb.config

    # local save directory
    savedir = 'data/embeddings/run-'+run.id
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # set seeds
    pl.seed_everything(42,workers=True)
    torch.set_float32_matmul_precision('high')

    # define data module
    data = SharpsDataModule(data_file=config.data['data_file'],
                           batch=config.training['batch_size'],
                           dim=config.data['dim'],
                           val_split=config.data['val_split'],
                           test=config.data['test'],
                           features=config.data['features'],
                           maxval=config.data['maxval'])

    # initialize model
    model = SharpEmbedder(config.model['base_channel_size'],
                          config.model['latent_dim'],
                          encoder_class = Encoder,
                          decoder_class = Decoder,
                          num_input_channels = 4,
                          image_size=config.data['dim'],
                          lambd=config.training['lambd'])
    
    # initialize wandb logger
    wandb_logger = WandbLogger(log_model='all')

    # select initial random subset of training data
    data.prepare_data()
    data.setup(stage='train')
    subset_files,_ = diverse_sampler(data.df_train['file'].tolist(),
                                     data.df_train[config.data['features']].to_numpy(),
                                     n=int(config.training['train_frac']*len(data.df_train)))

    # iterate training 
    for i in range(config.training['iterations']):
        # create training dataset from desired files
        data.subsample_trainset(subset_files)

        trainer = pl.Trainer(accelerator='gpu',
                            devices=1,
                            max_epochs=config.training['epochs'],
                            logger=wandb_logger,
                            deterministic=True,
                            precision=16)
        trainer.fit(model=model,train_dataloaders=data.subset_train_dataloader(),val_dataloaders=data.val_dataloader())

        # run inference on full training data 
        preds_train = trainer.predict(ckpt_path='best',dataloaders=data.train_dataloader(shuffle=False))
        files_train, embeddings_train, df_embed_train = save_predictions(preds_train,savedir,'train')
        
        # select diverse subset based on embedding pca unless on last iteration
        if i == config.training['iterations']-1:
            break

        pca = PCA(n_components=6,random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_train)
        subset_files,_ = diverse_sampler(files_train,embeddings_pca,n=int(config.training['train_frac']*len(files_train)))

    # run inference on validation set
    preds_val = trainer.predict(ckpt_path='best',model=model,dataloaders=data.val_dataloader())
    _,_,df_embed_val = save_predictions(preds_val,savedir,'val')

    # run inference on test set
    preds_test = trainer.predict(ckpt_path='best',model=model,dataloaders=data.test_dataloader())
    _,_,df_embed_test = save_predictions(preds_test,savedir,'test')

    # concatenate embeddings and merge with index data
    df_embeddings = pd.concat([df_embed_train,df_embed_val,df_embed_test])
    df_index = pd.read_csv(config.data['data_file'])
    df_embeddings = df_index.merge(df_embeddings,how='inner',on='file')
    data_file = savedir+'/embeddings.csv'
    df_embeddings.to_csv(data_file)

    # train LR for flare forecasting
    window = config.flareforecast['window']
    flare_thresh = config.flareforecast['flare_thresh']
    feats = ['embed'+str(i) for i in range(config.model['latent_dim'])] 
    train_frac = config.flareforecast['train_frac']

    results = {}
    metrics = []
    for val_split in range(5):
        model = LinearModel(data_file=data_file,window=window,flare_thresh=flare_thresh,
                            val_split=val_split,features=feats,max_iter=200)
        model.prepare_data()
        model.setup()
        if train_frac != 1:
            subsample_files,_ = diverse_sampler(model.df_train['file'].to_list(),
                                                model.X_train,
                                                n=int(train_frac*len(model.df_train)))
            model.subsample_trainset(subsample_files)
        model.train()
        ypred = model.test(model.X_pseudotest,model.df_pseudotest['flare'])
        y = model.df_pseudotest['flare']
        results['ypred'+str(val_split)] = ypred
        results['ytrue'] = y
        metrics.append(print_metrics(ypred,y))

    df_results = pd.DataFrame(results)
    df_results.insert(0,'filename',model.df_pseudotest['file'])
    df_results['ypred_median'] = df_results['ypred_median'] = df_results.filter(regex='ypred[0-9]').median(axis=1)
    print('Ensemble median:')
    metrics.append(print_metrics(df_results['ypred_median'],df_results['ytrue']))

    metrics_table = wandb.Table(columns=['MSE','BSS','APS','Gini','TSS','HSS','TPR','FPR'],data=metrics)
    wandb.log({'flare_forecast_metrics':metrics_table})
    
    wandb.finish()

if __name__ == "__main__":
    main()
