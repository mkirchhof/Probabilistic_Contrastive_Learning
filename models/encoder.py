import torch
from torch import nn
from utils.utils import construct_mlp

class Encoder(nn.Module):
    def __init__(self, n_hidden=2, dim_x=10, dim_z=2, dim_hidden=32,
                 post_kappa_min=20, post_kappa_max=80, x_samples=None,
                 device=torch.device('cuda:0'), has_joint_backbone=False):
        super().__init__()

        # Save parameters
        self.device = device
        self.post_kappa_min = torch.tensor(post_kappa_min, device=device)
        self.post_kappa_max = torch.tensor(post_kappa_max, device=device)
        self.dim_x = dim_x
        self.dim_z = dim_z

        # Create networks
        self.has_joint_backbone = has_joint_backbone
        self.mu_net = construct_mlp(n_hidden=n_hidden, dim_x=dim_x, dim_z=dim_z, dim_hidden=dim_hidden)
        self.kappa_net = construct_mlp(n_hidden=n_hidden, dim_x=dim_x if not has_joint_backbone else dim_z, dim_z=1, dim_hidden=dim_hidden)
        self.mu_net = self.mu_net.to(device)
        self.kappa_net = self.kappa_net.to(device)

        # Bring the kappa network to the correct range
        self.kappa_upscale = 1.
        self.kappa_add = 0.
        with torch.no_grad():
            self._rescale_kappa(x_samples)

        # Turn on gradients
        for p in self.mu_net.parameters():
            p.requires_grad = True
        for p in self.kappa_net.parameters():
            p.requires_grad = self.kappa_upscale.item() < float("Inf") # if we use infinite kappas, gradients break. So, turn off.

    def forward(self, x):
        # Return posterior (z-space) means and kappas for a batch of x
        mu = self.mu_net(x)
        mu = mu / torch.norm(mu, dim=-1).unsqueeze(-1)
        kappa = torch.exp(self.kappa_upscale * torch.log(1 + torch.exp(self.kappa_net(x if not self.has_joint_backbone else mu))) + self.kappa_add)
        return mu, kappa

    def _rescale_kappa(self, x_samples=None):
        # Goal: Find scale and shift parameters to bring the kappas to the desired range
        # indicated by self.post_kappa_min and self.post_kappa_max
        if torch.isinf(self.post_kappa_min) or torch.isinf(self.post_kappa_max):
            self.kappa_upscale = torch.ones(1, device=self.device) * float("inf")
            self.kappa_add = torch.ones(1, device=self.device) * float("inf")
        else:
            if self.post_kappa_max <= self.post_kappa_min:
                raise("post_kappa_max has to be > post_kappa_min.")
            if x_samples is None:
                raise("Please provide x_samples to the encoder to know which region of x we're dealing with.")
            kappa_samples = torch.log(1 + torch.exp(self.kappa_net(x_samples)))
            sample_min = torch.min(kappa_samples)
            sample_max = torch.max(kappa_samples)

            self.kappa_upscale = (torch.log(self.post_kappa_max) - torch.log(self.post_kappa_min)) / (
                         sample_max - sample_min)
            self.kappa_add = torch.log(self.post_kappa_max) - self.kappa_upscale * sample_max
