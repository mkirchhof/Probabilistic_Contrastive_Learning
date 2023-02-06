from torch import nn
import torch
from utils.vmf_sampler import VonMisesFisher
from utils.utils import pairwise_cos_sims, pairwise_l2_dists, log_vmf_norm_const


class MCInfoNCE(nn.Module):
    def __init__(self, kappa_init=20, n_samples=16, device=torch.device('cuda:0')):
        super().__init__()

        self.n_samples = n_samples
        self.kappa = torch.nn.Parameter(torch.ones(1, device=device) * kappa_init, requires_grad=True)

    def forward(self, mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg):
        # mu_neg and mu_pos is of dimension [batch, n_neg, dim]
        # mu_ref is dimension [batch, dim]
        mu_ref = mu_ref.unsqueeze(1)
        kappa_ref = kappa_ref.unsqueeze(1)

        # Draw samples (new dimension 0 contains the samples)
        samples_ref = VonMisesFisher(mu_ref, kappa_ref).rsample(self.n_samples) # [n_MC, batch, n_pos, dim]
        samples_pos = VonMisesFisher(mu_pos, kappa_pos).rsample(self.n_samples)
        if mu_neg is not None:
            samples_neg = VonMisesFisher(mu_neg, kappa_neg).rsample(self.n_samples)
        else:
            # If we don't get negative samples, treat the next batch sample as negative sample
            samples_neg = torch.roll(samples_pos, 1, 1)

        # calculate the standard log contrastive loss for each vmf sample
        negs = torch.logsumexp(torch.sum(samples_ref * samples_neg, dim=3) * self.kappa - torch.log(torch.ones(1).cuda() * samples_neg.shape[2]), dim=2)
        log_denominator_pos = torch.logsumexp(torch.stack((torch.sum(samples_ref * samples_pos, dim=3).squeeze(2) * self.kappa, negs), dim=0), dim=0)
        log_numerator_pos = torch.sum(samples_ref * samples_pos, dim=3) * self.kappa
        log_py1_pos = log_numerator_pos - log_denominator_pos.unsqueeze(2)

        # Average over the samples (we actually want a logmeanexp, that's why we substract log(n_samples))
        log_py1_pos = torch.logsumexp(log_py1_pos, dim=0) - torch.log(torch.ones(1, device=self.kappa.device) * self.n_samples)

        # Calculate loss
        loss = torch.mean(log_py1_pos)
        return -loss


class ELK(nn.Module):
    def __init__(self, kappa_init=20, device=torch.device('cuda:0')):
        super().__init__()

        self.kappa = torch.nn.Parameter(torch.ones(1, device=device) * kappa_init, requires_grad=True)

    def log_ppk_vmf_vec(self, mu1, kappa1, mu2, kappa2):
        p = mu1.shape[-1]

        kappa3 = torch.linalg.norm(kappa1 * mu1 + kappa2 * mu2, dim=-1).unsqueeze(-1)
        ppk = log_vmf_norm_const(kappa1, p) + log_vmf_norm_const(kappa2, p) - log_vmf_norm_const(kappa3, p)
        ppk = ppk * self.kappa

        return ppk.squeeze(-1)

    def forward(self, mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg):
        # mu_neg and mu_pos is of dimension [batch, n_neg, dim]
        # mu_ref is dimension [batch, dim]
        mu_ref = mu_ref.unsqueeze(1)
        kappa_ref = kappa_ref.unsqueeze(1)

        # Calculate similarities
        sim_pos = self.log_ppk_vmf_vec(mu_ref, kappa_ref, mu_pos, kappa_pos)
        if mu_neg is not None:
            sim_neg = self.log_ppk_vmf_vec(mu_ref, kappa_ref, mu_neg, kappa_neg)
        else:
            # If we don't get negative samples, treat the next batch sample as negative sample
            sim_neg = torch.roll(sim_pos, 1, 0)

        # Calculate loss
        loss = torch.mean(sim_pos, dim=1) - torch.logsumexp(torch.cat((sim_pos, sim_neg), dim=1), dim=1)
        loss = -torch.mean(loss)
        return loss


class HedgedInstance(nn.Module):
    def __init__(self, kappa_init=1, b_init=0, n_samples=16, device=torch.device('cuda:0')):
        super().__init__()

        self.n_samples = n_samples
        self.kappa = torch.nn.Parameter(torch.ones(1, device=device) * kappa_init, requires_grad=True) # kappa is "a" in the notation of their paper
        self.b = torch.nn.Parameter(torch.ones(1, device=device) * b_init, requires_grad=True)

    def forward(self, mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg):
        # mu_neg and mu_pos is of dimension [batch, n_neg, dim]
        # mu_ref is dimension [batch, dim]
        mu_ref = mu_ref.unsqueeze(1)
        kappa_ref = kappa_ref.unsqueeze(1)

        # Draw samples (new dimension 0 contains the samples)
        samples_ref = VonMisesFisher(mu_ref, kappa_ref).rsample(self.n_samples) # [n_MC, batch, n_pos, dim]
        samples_pos = VonMisesFisher(mu_pos, kappa_pos).rsample(self.n_samples)
        if mu_neg is not None:
            samples_neg = VonMisesFisher(mu_neg, kappa_neg).rsample(self.n_samples)
        else:
            # If we don't get negative samples, treat the next batch sample as negative sample
            samples_neg = torch.roll(samples_pos, 1, 1)

        # calculate the standard log contrastive loss for each vmf sample
        py1_pos = torch.sigmoid(self.kappa * torch.sum(samples_ref * samples_pos, dim=-1) + self.b)
        py1_neg = torch.sigmoid(self.kappa * torch.sum(samples_ref * samples_neg, dim=-1) + self.b)

        # Average over the samples
        log_py1_pos = torch.mean(torch.log(py1_pos), dim=0)
        log_py0_neg = torch.mean(torch.log(1 - py1_neg), dim=0)

        # Calculate loss
        loss = torch.mean(log_py1_pos) + torch.mean(log_py0_neg) / log_py0_neg.shape[-1]
        return -loss


def smoothness_loss(x, z):
    x_dist = pairwise_l2_dists(x)
    z_dist = 1 - pairwise_cos_sims(z)/ 2

    loss = torch.mean((x_dist - z_dist)**2 * (torch.sqrt(torch.ones(1, device=x.device) * 2) - z_dist.detach())**4)

    return loss
