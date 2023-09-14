import pytorch_lightning as pl
from torch import nn
import torch
# from pl_bolts.models.autoencoders.components import (
#     resnet18_decoder,
#     resnet18_encoder,
# )
from dataset import SHARPdataset
from enc_dec import ResNet18Enc, ResNet18Dec


class SharpVAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=20, wandb_logger: bool = True):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = ResNet18Enc(nc=4)
        self.decoder = ResNet18Dec(z_dim=latent_dim,nc=4)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.wandb_logger = wandb_logger

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-1, 1)
            module.bias.data.fill_(0.0)
            
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'mse': -recon_loss.mean()/(128*128*4)
        })

        return elbo
if __name__=='__main__':
    data_path = '/d0/kvandersande/sharps_hdf5/'
    image_size = 128
    dataset = SHARPdataset(data_path,image_size=image_size)
    (image, filename) = dataset[1]
    model = SharpVAE()
    enc = model.encoder(image[None,:,:,:])
    mu = model.fc_mu(enc)
    print(image.shape)
    print(enc.shape)
    print(mu.shape)
    image_out = model.decoder(mu)
    print(image_out.shape)
    #print(model)