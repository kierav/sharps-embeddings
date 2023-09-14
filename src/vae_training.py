import torch
import pytorch_lightning as pl
from vae import SharpVAE
from dataset import SHARPdataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from vae_plotter import SharpVAEplotter
import wandb

def main():
    MODEL_DIR = '/d0/subhamoy/models/vae/sharps/'
    DATA_DIR = '/d0/kvandersande/sharps_hdf5/'
    checkpoint_callback = ModelCheckpoint(dirpath=MODEL_DIR,
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='elbo',
                                          mode='min')

    #os.environ["WANDB_MODE"] = "offline"
    pl.seed_everything(23)
    torch.set_float32_matmul_precision('high')
    
    bs = 64
    ldim = 20
    backbone = 'resnet18'
    stride = 1
        
    wandb_logger = WandbLogger(entity="sc8473",
                               # Set the project where this run will be logged
                               project="vae_sharp",
                               name = f'Subh_stride_{stride}_bs_{bs}_ld_{ldim}_{backbone}',
                               # Track hyperparameters and run metadata
                                config={
                                    "learning_rate": 1e-4,
                                    "epochs": 30,
                                    "batch_size": bs,
                                    "latent_dim": ldim,
                                    "backbone": backbone
                                })
    
    vae = SharpVAE()
    plotter = SharpVAEplotter()
    trainer = pl.Trainer(max_epochs=30,
                         accelerator='gpu',
                         devices=[0],
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, plotter])
    image_size = 128
    sharp_dataset = SHARPdataset(DATA_DIR,image_size=image_size)
    
    if stride>1:
        indices = range(0, len(sharp_dataset), stride)
        sharp_dataset = torch.utils.data.Subset(sharp_dataset, indices)
        
    dataLoader = torch.utils.data.DataLoader(sharp_dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=4) 
    trainer.fit(vae, dataLoader)
    wandb.finish()
if __name__=='__main__':
    main()