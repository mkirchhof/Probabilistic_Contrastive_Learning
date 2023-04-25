import matplotlib.pyplot as plt
from models.generator import Generator
from models.encoder import Encoder
from utils.losses import MCInfoNCE, ELK, HedgedInstance
from tqdm import tqdm
from arguments import get_args
from utils.utils import init_seeds, pairwise_cos_sims, init_wandb
from utils.metrics import numerical_eval, graphical_eval
import torch
import os
import json
from shutil import rmtree
import wandb
from torch.optim.lr_scheduler import StepLR

def train_loop(args, gen, enc, loss):
    with torch.no_grad():
        eval_set = gen._sample_x(args.n_numerical_eval)

    # train
    enc.train()
    loss.train()
    running_loss = 0
    # If n_phases == 0, means we want to do one long phase with 2 * args.n_batches_per_half_phase batches
    n_total_batches = max(args.n_phases, 1) * 2 * args.n_batches_per_half_phase
    for b in tqdm(range(n_total_batches), position=0, leave=True):
        # Choose which parameter training phase we are in
        if b % args.n_batches_per_half_phase == 0:
            if args.n_phases == 0:
                # Train parameters jointly
                # To avoid restarting the optimizer, do this only in the beginning
                if b == 0:
                    params = list(enc.mu_net.parameters()) + list(enc.kappa_net.parameters())
                    if args.l_learnable_params:
                        params += list(loss.parameters())
                    optim = torch.optim.Adam(params, lr=args.lr)
                    scheduler = StepLR(optim, step_size=args.n_batches_per_half_phase, gamma=args.lr_decrease_after_phase)
                    n_neg = args.n_neg
            else:
                # First train mu then kappa (then mu, then kappa, then mu...)
                lr = args.lr * args.lr_decrease_after_phase**(b / (2 * args.n_batches_per_half_phase))
                if (b / args.n_batches_per_half_phase) % 2 == 0:
                    # time to train mu:
                    params = list(enc.mu_net.parameters())
                    if args.l_learnable_params:
                        params += list(loss.parameters())
                    optim = torch.optim.Adam(params, lr=lr)
                    scheduler = StepLR(optim, step_size=args.n_batches_per_half_phase / 2, gamma=args.lr_decrease_after_phase)
                    # Use rolled negatives in the same batch when training mu in high dimensions (faster and no empirical difference)
                    n_neg = args.n_neg if args.g_dim_z == 2 else 0
                else:
                    # time to train kappa
                    params = list(enc.kappa_net.parameters())
                    if args.l_learnable_params:
                        params += list(loss.parameters())
                    optim = torch.optim.Adam(params, lr=lr)
                    scheduler = StepLR(optim, step_size=args.n_batches_per_half_phase / 2, gamma=args.lr_decrease_after_phase)
                    # Use repeated mu when training kappa
                    n_neg = args.n_neg

        # Train
        optim.zero_grad()
        # Use the generator to create a batch
        x_ref, x_pos, x_neg = gen.sample(n=args.bs, n_neg=n_neg, oversampling_factor=args.oversampling_factor)
        mu_ref, kappa_ref = enc(x_ref)
        mu_pos, kappa_pos = enc(x_pos)
        # If we have n_neg = 0, then do not try to forward it
        if x_neg is not None:
            mu_neg, kappa_neg = enc(x_neg)
        else:
            mu_neg = None
            kappa_neg = None
        # Calculate loss
        cur_loss = loss(mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg)
        cur_loss.backward()
        running_loss += cur_loss.detach().cpu().item()
        optim.step()
        scheduler.step()

        # Val
        if b == 0 or (b+1) % args.eval_every == 0 or b == n_total_batches - 1:
            if b == 0:
                avg_loss = running_loss
            else:
                avg_loss = running_loss / args.eval_every
            print(f'Loss: {avg_loss}')
            enc.eval()
            cosdist_mse, cosdist_corr, cosdist_rankcorr, kappa_mse, kappa_corr, kappa_rankcorr = numerical_eval(args, gen, enc, eval_set, eval_std_instead_of_param=args.eval_std_instead_of_param)
            if args.n_graphical_eval > 0:
                graphical_eval(args, gen, enc, eval_set[:args.n_graphical_eval], print_examples=b == n_total_batches - 1)
                plt.suptitle(f'After {b} batches: Loss={avg_loss:.4f}. Cosdist MSE={cosdist_mse:.3f} (cor={cosdist_corr:.3f}). Kappa MSE={kappa_mse:.3f} (cor={kappa_corr:.3f}).')
                plt.savefig(f'results/{args.savefolder}/embeds_after_{b:06d}_batches.png')
                plt.close()
            results_dict = {"cosdist_mse": cosdist_mse.detach().cpu().item(),
                            "cosdist_corr": cosdist_corr,
                            "cosdist_rankcorr": cosdist_rankcorr,
                            "kappa_mse": kappa_mse.detach().cpu().item(),
                            "kappa_corr": kappa_corr,
                            "kappa_rankcorr":kappa_rankcorr,
                            "temperature":loss.kappa.detach().cpu().item(),
                            "loss": avg_loss,
                            "batches": b}
            if args.use_wandb:
                wandb.log(results_dict)
            with open(f'results/{args.savefolder}/latest_results.json', 'w') as f2:
                json.dump(results_dict, f2)
            enc.train()
            running_loss = 0

    enc.eval()
    return enc

def create_well_conditioned_generator(args):
    gen = None
    spread = 1
    while spread > args.g_min_spread or gen is None:
        # Create a new generator candidate
        gen = Generator(dim_x=args.g_dim_x, dim_hidden=args.g_dim_hidden, dim_z=args.g_dim_z, n_hidden=args.g_n_hidden,
                        pos_kappa=args.g_pos_kappa, post_kappa_min=args.g_post_kappa_min, post_kappa_max=args.g_post_kappa_max,
                        family=args.g_post_family)

        # Measure how much space in the latent space it fills
        samples = gen._sample_x(1000)
        mus, _ = gen(samples)
        spread = torch.min(pairwise_cos_sims(mus))

    return gen

def get_loss(args):
    if args.loss == "MCInfoNCE":
        loss = MCInfoNCE(kappa_init=args.g_pos_kappa, n_samples=args.l_n_samples)
    elif args.loss == "ELK":
        loss = ELK(kappa_init=args.g_pos_kappa)
    elif args.loss == "HedgedInstance":
        loss = HedgedInstance(kappa_init=args.l_hib_a, n_samples=args.l_n_samples, b_init=args.l_hib_b)
    else:
        raise(f"loss {args.loss} is not implemented.")

    return loss

if __name__=="__main__":
    args = get_args()

    ################### SETUP ###################
    init_seeds(args.seed)
    loss = get_loss(args)
    gen = create_well_conditioned_generator(args)
    enc = Encoder(dim_x=args.g_dim_x, dim_hidden=args.e_dim_hidden, dim_z=args.e_dim_z, n_hidden=args.e_n_hidden,
                  post_kappa_min=args.e_post_kappa_min, post_kappa_max=args.e_post_kappa_max, x_samples=gen._sample_x(1000))

    # Clean up the output folder
    os.makedirs("results", exist_ok=True)
    rmtree(f'results/{args.savefolder}', ignore_errors=True)
    os.makedirs(f'results/{args.savefolder}', exist_ok=True)
    with open(f'results/{args.savefolder}/parameters.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    init_wandb(args)

    ################### TRAIN ###################
    train_loop(args, gen, enc, loss)

    print("Fin.")
