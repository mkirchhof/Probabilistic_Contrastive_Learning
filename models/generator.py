import torch
from torch import nn
from utils.utils import construct_mlp, vmf_norm_ratio
from utils.vmf_sampler import VonMisesFisher
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from utils.losses import smoothness_loss


def smoothen_via_training(gen, print_progress=False):
    # Turn on gradients
    for p in gen.mu_net.parameters():
        p.requires_grad = True
    gen.train()

    # train
    optim = torch.optim.Adam(gen.parameters(), lr=0.01)
    running_loss = 0
    for b in range(5000):
        optim.zero_grad()
        with torch.no_grad():
            x = gen._sample_x(64)
        x.requires_grad = True
        mu, _ = gen(x)
        loss = 0
        loss += smoothness_loss(x, mu)
        running_loss += loss.detach()
        loss.backward()
        optim.step()

        if print_progress and b % 500 == 0:
            if b == 0:
                avg_loss = running_loss
            else:
                avg_loss = running_loss / 500
            print(f'Loss: {avg_loss}')
            running_loss = 0

    # Turn off gradients
    gen.eval()
    for p in gen.mu_net.parameters():
        p.requires_grad = False

    return gen

class Generator(nn.Module):
    def __init__(self, n_hidden=2, dim_x=10, dim_z=2, dim_hidden=32, pos_kappa=10, n_samples=10,
                 post_kappa_min=20, post_kappa_max=80, family="vmf", device=torch.device('cuda:0'),
                 has_joint_backbone=False):
        super().__init__()

        # Save parameters
        self.device = device
        self.post_kappa_min = torch.tensor(post_kappa_min, device=device)
        self.post_kappa_max = torch.tensor(post_kappa_max, device=device)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.pos_kappa = pos_kappa
        self.family = family

        # For sampling
        self.n_samples = n_samples
        self.denom_const = None  # will be calculated on demand below

        # Create networks
        self.has_joint_backbone = has_joint_backbone
        self.mu_net = construct_mlp(n_hidden=n_hidden, dim_x=dim_x, dim_z=dim_z, dim_hidden=dim_hidden)
        self.kappa_net = construct_mlp(n_hidden=n_hidden - 1, dim_x=dim_x if not has_joint_backbone else dim_z, dim_z=1, dim_hidden=dim_hidden)
        self.mu_net = self.mu_net.to(device)
        self.kappa_net = self.kappa_net.to(device)

        # Turn off gradients
        for p in self.mu_net.parameters():
            p.requires_grad = False
        for p in self.kappa_net.parameters():
            p.requires_grad = False

        # Bring the kappa network to the correct range
        self.kappa_upscale = 1.
        self.kappa_add = 0.
        self._rescale_kappa()

        smoothen_via_training(self)

    def forward(self, x):
        # Return posterior (z-space) means and kappas for a batch of x
        mu = self.mu_net(x)
        mu = mu / torch.norm(mu, dim=-1).unsqueeze(-1)
        kappa = torch.exp(self.kappa_upscale * torch.log(1 + torch.exp(self.kappa_net(x if not self.has_joint_backbone else mu))) + self.kappa_add)
        return mu, kappa

    def _rescale_kappa(self):
        # Goal: Find scale and shift parameters to bring the kappas to the desired range
        # indicated by self.post_kappa_min and self.post_kappa_max
        if torch.isinf(self.post_kappa_min) or torch.isinf(self.post_kappa_max):
            self.kappa_upscale = torch.ones(1, device=self.device) * float("inf")
            self.kappa_add = torch.ones(1, device=self.device) * float("inf")
        else:
            if self.post_kappa_max <= self.post_kappa_min:
                raise("post_kappa_max has to be > post_kappa_min.")
            x_samples = self._sample_x(1000)
            kappa_samples = torch.log(1 + torch.exp(self.kappa_net(x_samples)))
            sample_min = torch.min(kappa_samples)
            sample_max = torch.max(kappa_samples)

            self.kappa_upscale = (torch.log(self.post_kappa_max) - torch.log(self.post_kappa_min)) / (
                         sample_max - sample_min)
            self.kappa_add = torch.log(self.post_kappa_max) - self.kappa_upscale * sample_max

    def sample(self,n=64, n_neg=1, oversampling_factor=1, same_ref=False):
        # Generates (x_ref, x_pos, x_neg) triplets.
        # Input:
        # n - integer, batchsize (number of x_ref)
        # n_neg - integer, number of negatives (0 to return None)
        # oversampling_factor - integer, how many candidates to generate to select x_pos and x_neg from.
        #                       Use a value as high as possible, otherwise need to resample
        # same_ref - boolean, whether to use the same x_ref for the whole batch (for debugging)

        # Generate random samples from x-space
        x_ref = self._sample_x(n)
        if same_ref:
            x_ref[:,:] = x_ref[0,:]
        z_ref = self._sample_z_from_x(x_ref)

        # generate pos and neg samples to the above samples
        x_pos, x_neg, _, _ = self._sample_pos_neg_by_candidates(z_ref, n_neg, oversampling_factor)

        return x_ref, x_pos, x_neg

    def _sample_x(self, n):
        return torch.rand((n, self.dim_x), device=self.device)

    def _sample_z_from_x(self, x):
        # Takes a batch of x, encodes their posteriors and draws from them
        mu, kappa = self.forward(x)
        if self.family == "vmf":
            z_distrs = VonMisesFisher(mu, kappa)
        elif self.family == "Gaussian":
            z_distrs = Normal(mu, 1/torch.sqrt(kappa))
        elif self.family == "Laplace":
            z_distrs = Laplace(mu, 1/kappa)
        z_samples = z_distrs.sample()
        z_samples = torch.nn.functional.normalize(z_samples, dim=-1)
        return z_samples

    def _sample_pos_neg_by_candidates(self, z_ref, n_neg=1, oversampling_factor=1):
        # Sample x-candidates, encode them into z and try to find pos/neg matches to the reference points
        # Works if the area that z_pos covers inside the whole z space is relatively high.
        # z_ref - [batchsize, x_dim] batch of reference points
        # oversampling_factor - integer, how many candidates to generate to select x_pos and x_neg from.
        #                       Use a value as high as possible, otherwise need to resample

        x_pos, z_pos = self._sample_candidates(z_ref, n=1, want_pos=True, oversampling_factor=oversampling_factor)
        if n_neg > 0:
            x_neg, z_neg = self._sample_candidates(z_ref, n=n_neg, want_pos=False, oversampling_factor=oversampling_factor)
        else:
            x_neg = None
            z_neg = None

        return x_pos, x_neg, z_pos, z_neg

    def _sample_candidates(self, z_ref, n=1, want_pos=True, oversampling_factor=1):
        batchsize = z_ref.shape[0]

        # Generate candidates until each z_ref has a sample
        x_partner = torch.zeros((batchsize, n, self.dim_x), device=self.device)
        z_partner = torch.zeros((batchsize, n, self.dim_z), device=self.device)
        needs_partner = torch.ones((batchsize, n), dtype=torch.uint8, device=self.device)
        while torch.any(needs_partner):
            requires_partner = torch.any(needs_partner, dim=1)
            n_require_partner = torch.sum(requires_partner)
            x_cand = self._sample_x(n_require_partner * n * oversampling_factor)
            z_cand = self._sample_z_from_x(x_cand)

            # sample whether the candidates are pos/neg to the ref
            # Each x_ref has its own candidates
            cand_per_ref = z_cand.reshape(n_require_partner, n*oversampling_factor, z_cand.shape[-1])
            prob_ref_and_cand_pos = self._pos_prob(z_ref[requires_partner].unsqueeze(1), cand_per_ref)
            is_ref_and_cand_pos = torch.bernoulli(prob_ref_and_cand_pos)
            is_ref_and_cand_pos = is_ref_and_cand_pos.type(torch.uint8)
            is_ref_and_cand_wanted = is_ref_and_cand_pos == want_pos

            # Choose samples
            # in is_ref_and_cand_wanted we might have rows with full 0. This crashes torch.multinomial.
            # In case we have no 1, give everything a one and then filter out everything again afterwards
            p_select_bigger0 = is_ref_and_cand_wanted.float() + (torch.sum(is_ref_and_cand_wanted, dim=1) == 0).unsqueeze(1)
            chosen_idxes = torch.multinomial(p_select_bigger0, n, replacement=False)
            # Currently, chosen_idxes indices the columns per row.
            # We want to get back to the original indexing of the flattened x_cand and z_cand tensors:
            chosen_idxes = chosen_idxes + torch.arange(n_require_partner, device=chosen_idxes.device).unsqueeze(1) * n * oversampling_factor

            if n > 1:
                # If we need several samples, we need to fill in the tensor sample by sample, because we might have
                # a different amount of valid candidates per sample and cannot tensorize this indexing
                for sub_idx, overall_idx in enumerate(requires_partner.nonzero()[:,0]):
                    # sub_idx is the index with respect to those that require a partner (the first that requires a partner, the second, ...)
                    # overall_idx is the general idx of those samples (e.g., 8, 17, 52, ...)
                    # The chosen_idx will probably contain samples with probability 0, because we forced it to sample n things,
                    # even if there were less than n possible 1s in the array.
                    n_matches = torch.sum(is_ref_and_cand_wanted[sub_idx])
                    n_needed = torch.sum(needs_partner[overall_idx, :])
                    n_new_samples = torch.min(n_matches, n_needed).type(torch.int)
                    if n_new_samples > 0:
                        # One trick we can use is that the prob-0 choices are always at the end
                        chosen_idx = chosen_idxes[sub_idx,:n_new_samples]
                        x_partner[overall_idx, n - n_needed:(n - n_needed + n_new_samples)] = x_cand[chosen_idx, :]
                        z_partner[overall_idx, n - n_needed:(n - n_needed + n_new_samples)] = z_cand[chosen_idx, :]
                        needs_partner[overall_idx, n - n_needed:(n - n_needed + n_new_samples)] = False
            elif n == 1:
                # We can speed up the indexing by tensorizing it
                n_matches = torch.sum(is_ref_and_cand_wanted, dim=1)
                x_partner[requires_partner.nonzero()[n_matches > 0, 0], 0] = x_cand[chosen_idxes[n_matches > 0, 0], :]
                z_partner[requires_partner.nonzero()[n_matches > 0, 0], 0] = z_cand[chosen_idxes[n_matches > 0, 0], :]
                needs_partner[requires_partner.nonzero()[n_matches > 0, 0], 0] = False

        return x_partner, z_partner

    def _pos_prob(self, z1, z2):
        # Returns P(Y = 1|z_1, z_2) based on the P(z_2|Y=1, z_1) pos-vMF distribution
        # and the uniform distribution for negative samples
        # Input:
        #  z_1 - [batchsize_1, z_dim] tensor containing rowwise normalized zs
        #  z_2 - [batchsize_2, z_dim] tensor containing rowwise normalized zs
        # Output:
        #  [batchsize_1, batchsize_2] tensor containing probabilities P(Y=1) in [0, 1]

        # Calculate these constants here and not in the class init, because not all strategies need them
        if self.denom_const is None:
            self.denom_const = torch.tensor(vmf_norm_ratio(self.pos_kappa, self.dim_z), device=self.device)

        cos = torch.sum(z1 * z2, dim=-1)
        log_pos_dens = self.pos_kappa * cos
        log_neg_dens = self.denom_const

        return torch.exp(log_pos_dens - torch.logsumexp(torch.stack((log_pos_dens, log_neg_dens * torch.ones(log_pos_dens.shape, device=self.device)), dim=0), dim=0))
