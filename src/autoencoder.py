import matplotlib.pyplot as plt
import pytorch_lightning as pl
import random
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from IPython.display import set_matplotlib_formats
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torchmetrics
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.notebook import tqdm
from dataset import SHARPdataset
from utils import plot_reconstruction

import wandb

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("Device:", device)
class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int,
                 latent_dim: int, act_fn: object = nn.ReLU,
                 image_size: int = 256):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. 
                                Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * ((image_size//8)**2) * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int,
                 latent_dim: int, act_fn: object = nn.ReLU,
                 image_size: int = 256):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers.
                                Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.image_size = image_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * ((image_size//8)**2) * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Sigmoid(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, (self.image_size//8), (self.image_size//8))
        x = self.net(x)
        return x


class SharpEmbedder(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        image_size: int = 256,
        wandb_logger: bool = True,
        loss_type: str = 'mse',
        lambd: float = 1e-3
    ):
        super().__init__()
        self.latent_dim = latent_dim
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim, image_size=image_size)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim, image_size=image_size)
        # Example input array needed for visualizing the graph of the network
        self.wandb_logger = wandb_logger
        self.loss_type = loss_type
        self.lambd = lambd
        # define metrics
        self.val_r2 = torchmetrics.R2Score(num_outputs=20)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z,x_hat

    def _get_reconstruction_loss(self, x, f, x_hat, z):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""

        loss1 = F.mse_loss(x, x_hat, reduction="none")
        loss1 = loss1.sum(dim=[1, 2, 3]).mean(dim=[0])

        # calculate loss between latent dim and features
        loss2 = F.mse_loss(f,z[:,:f.shape[1]],reduction='none')
        loss2 = loss2.sum(dim=[1]).mean(dim=[0])

        # return loss
        if self.loss_type == 'mse':
            loss = loss1
        elif self.loss_type == 'embed+feat':
            loss = loss1 + self.lambd*loss2
        else:
            loss = loss1

        return loss,loss1,loss2

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        _,x,f = batch  # We do not need the labels
        z,x_hat = self.forward(x)
        loss,loss1,loss2 = self._get_reconstruction_loss(x,f,x_hat,z)

        self.log_dict({"train_loss":loss,
                       'train_reconstruction_loss':loss1,
                       'train_sharps_loss':loss2})

        return loss

    def validation_step(self, batch, batch_idx):
        files,x,f = batch 
        z,x_hat = self.forward(x)
        loss,loss1,loss2 = self._get_reconstruction_loss(x,f,x_hat,z)
        
        totusflux_err = torch.abs(x_hat[:,3,:,:]).sum(dim=[1,2]).mean(dim=[0])-torch.abs(x[:,3,:,:]).sum(dim=[1,2]).mean(dim=[0])
        self.val_r2(z[:,:f.shape[1]],f.view(f.shape[0],-1))

        self.log_dict({"val_loss":loss,
                       'val_reconstruction_loss':loss1,
                       'val_sharps_loss':loss2,
                       'val_totusflux_err':totusflux_err,
                       'val_sharps_r2':self.val_r2})
        # log sample reconstruction
        if batch_idx == 0 & self.wandb_logger:
            idx = -1
            _,x_hat = self.forward(x)
            fig = plot_reconstruction(x[idx].detach().cpu().numpy(),
                                      x_hat[idx].detach().cpu().numpy(),
                                      self.latent_dim,
                                      files[idx])
            wandb.log({'sample_validation_img':fig})

    def test_step(self, batch, batch_idx):
        loss,totusflux_err = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
        self.log("test_totusflux_err", totusflux_err)

    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        """Given a batch of images, return embeddings from encoder"""
        f,x,_ = batch
        embedding = self.encoder(x)
        return f,embedding
    


if __name__=='__main__':
    data_path = '/d0/kvandersande/sharps_hdf5/'
    image_size = 256
    dataset = SHARPdataset(data_path,image_size=image_size)
    (image, filename) = dataset[1]
    model = SharpEmbedder(16, 20, encoder_class = Encoder,
                          decoder_class = Decoder,
                          num_input_channels = 4,
                          image_size=image_size)
    image_out = model(image)
    print(image.shape)
    print(image_out.shape)
    #print(model)
