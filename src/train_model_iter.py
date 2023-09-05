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
                          image_size=config.data['dim'])
    
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
                            log_every_n_steps=50,
                            logger=wandb_logger,
                            deterministic=True,
                            precision=16)
        trainer.fit(model=model,train_dataloaders=data.subset_train_dataloader(),val_dataloaders=data.val_dataloader())

        # run inference on full training data 
        preds_train = trainer.predict(ckpt_path='best',dataloaders=data.train_dataloader(shuffle=False))
        files_train, embeddings_train = save_predictions(preds_train,savedir,'train')
        
        # select diverse subset based on embedding pca unless on last iteration
        if i == config.training['iterations']-1:
            break

        pca = PCA(n_components=6,random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_train)
        subset_files,_ = diverse_sampler(files_train,embeddings_pca,n=int(config.training['train_frac']*len(files_train)))


    preds_val = trainer.predict(ckpt_path='best',model=model,dataloaders=data.val_dataloader())
    files_val, embeddings_val = save_predictions(preds_val,savedir,'val')

    preds_test = trainer.predict(ckpt_path='best',model=model,dataloaders=data.test_dataloader())
    files_test, embeddings_test = save_predictions(preds_test,savedir,'test')

    wandb.finish()

if __name__ == "__main__":
    main()
