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
    SHARP_PARAMS = ['usflux', 'meangam',
                    'meangbt', 'meangbz', 
                    'meangbh', 'meanjzd', 
                    'meanjzh', 'totusjz',
                    'totusjh', 'meanalp', 
                    'absnjzh', 'savncpp',
                    'meanpot', 'totpot', 
                    'meanshr', 'shrgt45',
                    'r_value', 'size',
                    'area', 'nacr',
                    'size_acr', 'area_acr',
                    'mtot', 'mnet',
                    'mpos_tot', 'mneg_tot',
                    'mmean', 'mstdev', 'mskew']


    #os.environ["WANDB_MODE"] = "offline"
    pl.seed_everything(23)
    torch.set_float32_matmul_precision('high')
    
    bs = 64
    ldim = 20
    backbone = 'resnet'
    stride = 1
    lr = 1e-4
    epochs = 100
    crop = False
    include_sharp = False
    gpu_number = 0
    flux_w = 10
    beta = 80
    name = f'Subh_stride_{stride}_bs_{bs}_ld_{ldim}_backbone_{backbone}_lr_{lr}_crop_{crop}_sharp_constr_{include_sharp}_beta_{beta}'
    
    
    checkpoint_callback = ModelCheckpoint(dirpath=MODEL_DIR,
                                        filename=name, #'{epoch}-{name}',
                                        save_top_k=1,
                                        verbose=True,
                                        monitor='elbo',
                                        mode='min')
        
    wandb_logger = WandbLogger(entity="sc8473",
                               # Set the project where this run will be logged
                               project="vae_sharp",
                               name = name,
                               # Track hyperparameters and run metadata
                                config={
                                    "learning_rate": lr,
                                    "epochs": epochs,
                                    "batch_size": bs,
                                    "latent_dim": ldim,
                                    "backbone": backbone
                                    #"flux_weight": flux_w
                                })
    
    vae = SharpVAE(latent_dim=ldim,
                   lr=lr, flux_w=flux_w,
                   include_sharp=include_sharp,
                   sharp_params=SHARP_PARAMS,
                   backbone=backbone)
    plotter = SharpVAEplotter(crop=crop,
                              include_sharp=include_sharp)
    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator='gpu',
                         devices=[gpu_number],
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, plotter])
    image_size = 128
    sharp_dataset = SHARPdataset(DATA_DIR, image_size=image_size,
                                 crop=crop, include_sharp=include_sharp,
                                 data_stride=1)
    
    # if stride>1:
    #     indices = range(0, len(sharp_dataset), stride)
    #     sharp_dataset = torch.utils.data.Subset(sharp_dataset, indices)
        
    dataLoader = torch.utils.data.DataLoader(sharp_dataset,
                                             batch_size=bs,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=4) 
    trainer.fit(vae, dataLoader)
    wandb.finish()
if __name__=='__main__':
    main()