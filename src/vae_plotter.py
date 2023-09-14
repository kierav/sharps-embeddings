from matplotlib.pyplot import imshow, figure
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

class SharpVAEplotter(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = 128
        self.num_preds = 1

    def on_train_epoch_end(self, trainer, pl_module):
        figure(figsize=(8, 3), dpi=300)

        # Z COMES FROM NORMAL(0, 1)
        rand_v = torch.rand((self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
        p = torch.distributions.Normal(torch.zeros_like(rand_v), torch.ones_like(rand_v))
        z = p.rsample()

        # SAMPLE IMAGES
        with torch.no_grad():
            image = pl_module.decoder(z.to(pl_module.device)).cpu()
            
        delta = 0.0
        fig, axes = plt.subplots(1,4, figsize=(8,2), constrained_layout=True)
        ax = axes.ravel()
        titles = ['Br', 'Bp', 'Bt', 'Blos']
        for i, t in enumerate(titles):
            im = ax[i].imshow(image[0,i,:,:],
                    cmap='gray',
                    vmin=delta,
                    vmax=1 - delta)
            ax[i].axis('off')
            ax[i].set_title(t)
            
        ticks = [delta, 0.5, 1-delta]
        tlabels = [f'{np.round(2000*(t-0.5))} G' for t in ticks]
        cbar = fig.colorbar(im, ticks=ticks)
        cbar.ax.set_yticklabels(tlabels) 
        plt.suptitle('generated')

        wandb.log({"Generated SHARP": wandb.Image(fig)})
        