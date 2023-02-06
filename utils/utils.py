import torch
from torch import nn
import numpy as np
from numpy import random
from numpy import i0  # modified Bessel function of first kind order 0, I_0
from scipy.special import ive  # exponential modified Bessel function of first kind, I_v * exp(-abs(kappa))
import wandb
import os

def init_seeds(seed=123):
    torch.backends.cudnn.deterministic = True;
    np.random.seed(seed);
    random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)

def construct_mlp(n_hidden=2, dim_x=10, dim_z=2, dim_hidden=32):
    if dim_hidden == 0:
        # Zimmermann setup
        dim_first = 10 * dim_z
        dim_middle = 50 * dim_z
        dim_last = 10 * dim_z
    else:
        dim_first = dim_hidden
        dim_middle = dim_hidden
        dim_last = dim_hidden

    layers = []
    layers.append(nn.Linear(dim_x, dim_first))
    prev_dim = dim_first
    for i in range(n_hidden - 1):
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(prev_dim, dim_middle))
        prev_dim = dim_middle
    if n_hidden - 1 >= 0:
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(prev_dim, dim_last))
        prev_dim = dim_last
    layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(prev_dim, dim_z))

    return nn.Sequential(*layers)

def _vmf_normalize(kappa, dim):
    """Compute normalization constant using built-in numpy/scipy Bessel
    approximations.
    Works well on small kappa and mu.
    Imported from https://github.com/jasonlaska/spherecluster/blob/develop/spherecluster/von_mises_fisher_mixture.py
    """
    if kappa < 1e-15:
        kappa = 1e-15

    num = (dim / 2.0 - 1.0) * np.log(kappa)

    if dim / 2.0 - 1.0 < 1e-15:
        denom = (dim / 2.0) * np.log(2.0 * np.pi) + np.log(i0(kappa))
    else:
        denom = (dim / 2.0) * np.log(2.0 * np.pi) + np.log(ive(dim / 2.0 - 1.0, kappa)) + kappa

    if np.isinf(num):
        raise ValueError("VMF scaling numerator was inf.")

    if np.isinf(denom):
        raise ValueError("VMF scaling denominator was inf.")

    const = np.exp(num - denom)

    if const == 0:
        raise ValueError("VMF norm const was 0.")

    return const

def vmf_norm_ratio(kappa, dim):
    # Approximates log(norm_const(0) / norm_const(kappa)) of a vMF distribution
    # See approx_vmf_norm_const.R to see how it was approximated

    if dim==2:
        return -2.439 + 0.9904 * kappa + 2.185e-4 * kappa**1.55
    elif dim==4:
        return -4.817 + 0.9713 * kappa + 6.479e-4 * kappa**1.55
    elif dim==8:
        return -7.908 + 0.9344 * kappa + 1.477e-3 * kappa**1.55
    elif dim==10:
        return -9.024 + 0.9165 * kappa + 1.877e-3 * kappa**1.55
    elif dim==12:
        return -9.958 + 0.8990 * kappa + 2.267e-3 * kappa**1.55
    elif dim==16:
        return -11.43 + 0.8649 * kappa + 3.020e-3 * kappa**1.55
    elif dim==32:
        return -14.38 + 0.7416 * kappa + 5.686e-3 * kappa**1.55
    elif dim==40:
        return -14.92 + 0.6868 * kappa + 6.837e-3 * kappa**1.55
    elif dim==48:
        return -15.13 + 0.6360 * kappa + 7.882e-3 * kappa**1.55
    elif dim==56:
        return -15.12 + 0.5889 * kappa + 8.833e-3 * kappa**1.55
    elif dim==64:
        return -14.94 + 0.5450 * kappa + 0.009698 * kappa**1.55
    elif dim==96:
        return -13.42 + 0.3973 * kappa + 1.246e-2 * kappa**1.55
    elif dim==128:
        return -11.44 + 0.2839 * kappa + 1.425e-2 * kappa**1.55
    elif dim==256:
        return -4.7340339 + 0.0289469 * kappa + 0.0173026 * kappa**1.55
    elif dim==512:
        return 0.8674 - 0.1124 * kappa + 0.01589 * kappa**1.55
    else:
        return np.log(_vmf_normalize(0, dim)) - np.log(_vmf_normalize(kappa, dim))

def log_vmf_norm_const(kappa, dim=10):
    # Approximates the log vMF normalization constant (for the ELK loss)
    # See approx_vmf_norm_const.R to see how it was approximated

    if dim==4:
        return -0.826604 - 0.354357 * kappa - 0.383723 * kappa**1.1
    if dim==8:
        return -1.29737 + 0.36841 * kappa - 0.80936 * kappa**1.1
    elif dim==10:
        return -1.27184 + 0.67365 * kappa - 0.98726 * kappa**1.1
    elif dim==16:
        return -0.23773 + 1.39146 * kappa - 1.39819 * kappa**1.1
    elif dim==32:
        return 8.07579 + 2.28954 * kappa - 1.86925 * kappa**1.1
    elif dim==64:
        return 38.82967 + 2.34269 * kappa - 1.77425 * kappa**1.1
    else:
        return np.log(_vmf_normalize(kappa, dim))

def pairwise_cos_sims(z):
    # Calculate pairwise cosine distances between rows in z return as flat vector
    cos_dists = torch.matmul(z, z.t())
    cos_dists = cos_dists[torch.tril_indices(cos_dists.shape[0], cos_dists.shape[1]).unbind()]
    return cos_dists

def pairwise_l2_dists(z):
    l2_dists = torch.cdist(z, z)
    l2_dists = l2_dists[torch.tril_indices(l2_dists.shape[0], l2_dists.shape[1]).unbind()]
    return l2_dists

def init_wandb(args):
    ### If wandb-logging is turned on, initialize the wandb-run here:
    if args.use_wandb:
        import re
        if args.wandb_key != "":
            _ = os.system('wandb login {}'.format(args.wandb_key))
            os.environ['WANDB_API_KEY'] = args.wandb_key
        else:
            print("No wandb key provided. Hoping that one was specified as environment variable.")
        # For the groupname, remove the seed, so that we can group per seed.
        group = re.sub("_seed_[^_.]*", "", args.savefolder)
        wandb.init(project=args.wandb_project, group=group, name=args.savefolder, dir=f"results/{args.savefolder}/")
        wandb.init(settings=wandb.Settings(start_method='fork'))
        wandb.config.update(args)
