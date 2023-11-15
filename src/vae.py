import pytorch_lightning as pl
from torch import nn
import torch
# from pl_bolts.models.autoencoders.components import (
#     resnet18_decoder,
#     resnet18_encoder,
# )
from dataset import SHARPdataset
from enc_dec import ResNet18Enc, ResNet18Dec, SimEnc, SimDec


class SharpVAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=20,
                 lr = 1e-4, flux_w = 10,
                 include_sharp: bool = False,
                 sharp_params = [],
                 wandb_logger: bool = True,
                 backbone = 'resnet'):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.backbone = backbone
        self.sharp_params = sharp_params
        
        if self.backbone=='resnet':
            self.encoder = ResNet18Enc(nc=4)
            self.decoder = ResNet18Dec(z_dim=latent_dim,nc=4)
        else:
            self.encoder = SimEnc(latent_dim=latent_dim)
            self.decoder = SimDec(latent_dim=latent_dim)
           
        
        self.sharp_enc = nn.Linear(len(self.sharp_params), latent_dim)
        self.sharp_dec = nn.Linear(latent_dim, len(self.sharp_params))
        
        self.fc_in = nn.Linear(enc_out_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, enc_out_dim)
        self.include_sharp = include_sharp
        
        
        if self.include_sharp:
            # distribution parameters
            self.fc_mu = nn.Linear(2*latent_dim, latent_dim)
            self.fc_var = nn.Linear(2*latent_dim, latent_dim)
        elif self.backbone=='resnet':
            # distribution parameters
            self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
            self.fc_var = nn.Linear(enc_out_dim, latent_dim)
        else:
            self.fc_mu = nn.Linear(latent_dim, latent_dim)
            self.fc_var = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU())
            

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.wandb_logger = wandb_logger
        self.lr = lr
        self.flux_w = flux_w

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
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
        

        # encode x to get the mu and variance parameters
        if self.include_sharp:
            x, p, _ = batch
            x1 = self.encoder(x)
            x1 = self.fc_in(x1)
            x2 = self.sharp_enc(p)
            x_encoded = torch.cat((x1, x2), 1)
            x_encoded = torch.sigmoid(x_encoded)           
            
        else:
            x, _ = batch
            x_encoded = self.encoder(x)
        
        
        
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        
        
        # decoded
        x_hat = self.decoder(z)
        if self.include_sharp:
            z1 = torch.sigmoid(z)
            p_hat = self.sharp_dec(z1)
        
        x_p = (x - 0.5)*((x>0.5).float())
        x_n = (x - 0.5)*((x<0.5).float())
        x_h_p = (x_hat - 0.5)*((x_hat>0.5).float())
        x_h_n = (x_hat - 0.5)*((x_hat<0.5).float())
        p_flux_diff = (x_p.sum(dim=[1,2,3]) - x_h_p.sum(dim=[1,2,3]))**2
        n_flux_diff = (x_n.sum(dim=[1,2,3]) - x_h_n.sum(dim=[1,2,3]))**2
        p_flux_diff = p_flux_diff/(4*128*128)
        n_flux_diff = n_flux_diff/(4*128*128)
        if self.include_sharp:
            sharp_diff = (p - p_hat)**2
            sharp_diff = sharp_diff.sum(dim=[1])/len(self.sharp_params)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x) 

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        if self.include_sharp:
            elbo = (kl - recon_loss + sharp_diff) # + self.flux_w*p_flux_diff + self.flux_w*n_flux_diff)
        else:
            elbo = (80*kl - recon_loss) # + self.flux_w*p_flux_diff + self.flux_w*n_flux_diff)
            
        elbo = elbo.mean()
        var = (torch.exp(self.log_scale)**2)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        
        if self.include_sharp:
            lg_dict = {
                'elbo': elbo,
                'kl': kl.mean(),
                'mse': -(recon_loss.mean()*2*var/(128*128*4)) - var*torch.log(2*torch.pi*var),
                'log_std': self.log_scale,
                'p_flux_diff': p_flux_diff.mean(),
                'n_flux_diff': n_flux_diff.mean(),
                'sharp_diff': sharp_diff.mean()
            }
        else:
            lg_dict = {
                'elbo': elbo,
                'kl': kl.mean(),
                'mse': -(recon_loss.mean()*2*var/(128*128*4)) - var*torch.log(2*torch.pi*var),
                'log_std': self.log_scale,
                'p_flux_diff': p_flux_diff.mean(),
                'n_flux_diff': n_flux_diff.mean()
            }
        self.log_dict(lg_dict)

        return elbo
if __name__=='__main__':
    data_path = '/d0/kvandersande/sharps_hdf5/'
    image_size = 128
    dataset = SHARPdataset(data_path,image_size=image_size)
    (image, filename) = dataset[1]
    model = SharpVAE(backbone='simple')
    enc = model.encoder(image[None,:,:,:])
    mu = model.fc_mu(enc)
    print(image.shape)
    print(enc.shape)
    print(mu.shape)
    image_out = model.decoder(mu)
    print(image_out.shape)
    #print(model)