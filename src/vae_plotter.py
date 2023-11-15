from matplotlib.pyplot import imshow, figure
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from dataset import SHARPdataset

class SharpVAEplotter(pl.Callback):
    def __init__(self,
                 include_sharp: bool = False,
                 crop: bool = False):
        super().__init__()
        self.img_size = 128
        self.num_preds = 1
        self.include_sharp = include_sharp
        self.crop = crop

    def on_train_epoch_end(self, trainer, pl_module):
        figure(figsize=(8, 3), dpi=300)

        # Z COMES FROM NORMAL(0, 1)
        rand_v = torch.rand((self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
        p = torch.distributions.Normal(torch.zeros_like(rand_v), torch.ones_like(rand_v))

        DATA_PATH = '/d0/kvandersande/sharps_hdf5/'
        if self.include_sharp:
            dataset = SHARPdataset(DATA_PATH,
                                image_size=self.img_size,
                                crop=self.crop, include_sharp=True)
            (image, p, _) = dataset[1]
            image_ = image[None,:,:,:]
            p_ = p[None,:]
        else:
            dataset = SHARPdataset(DATA_PATH,
                                image_size=self.img_size,
                                crop=self.crop)
            (image, _) = dataset[1]
            image_ = image[None,:,:,:]
        with torch.no_grad():
            enc = pl_module.encoder(image_.to(pl_module.device)).cpu()
            
            if self.include_sharp:
                enc = pl_module.fc_in(enc.to(pl_module.device)).cpu()
                p_enc = pl_module.sharp_enc(p_.to(pl_module.device)).cpu()
                enc = torch.cat((enc.to(pl_module.device), p_enc.to(pl_module.device)), 1).cpu()
                enc = torch.sigmoid(enc.to(pl_module.device)).cpu()        
                
            latent = pl_module.fc_mu(enc.to(pl_module.device)).cpu()
            latent_var = pl_module.fc_var(enc.to(pl_module.device)).cpu()
            std = torch.exp(latent_var / 2)
            q = torch.distributions.Normal(latent, std)
            z = q.rsample()
            image_out = pl_module.decoder(z.to(pl_module.device)).cpu()
        
        #z = p.rsample()

        # # SAMPLE IMAGES
        # with torch.no_grad():
        #     image = pl_module.decoder(z.to(pl_module.device)).cpu()
            
        delta = 0.0
        fig, axes = plt.subplots(2,4, figsize=(8,4), constrained_layout=True)
        ax = axes.ravel()
        titles = ['Br', 'Bp', 'Bt', 'Blos']
        for i, t in enumerate(titles):
            im = ax[i].imshow(image_[0,i,:,:],
                    cmap='gray',
                    vmin=delta,
                    vmax=1 - delta)
            ax[i].axis('off')
            ax[i].set_title(t)
            
        for i, t in enumerate(titles):
            im = ax[4+i].imshow(image_out[0,i,:,:],
                    cmap='gray',
                    vmin=delta,
                    vmax=1 - delta)
            ax[4+i].axis('off')
            # ax[4+i].set_title(t)
            
        ticks = [delta, 0.5, 1-delta]
        tlabels = [f'{np.round(2000*(t-0.5))} G' for t in ticks]
        cbar = fig.colorbar(im, ticks=ticks)
        cbar.ax.set_yticklabels(tlabels) 
        plt.suptitle('generated')

        wandb.log({"Generated SHARP": wandb.Image(fig)})
        