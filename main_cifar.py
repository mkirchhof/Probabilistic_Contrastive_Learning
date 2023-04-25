from models.encoder_resnet import ResnetProbEncoder
from tqdm import tqdm
from arguments import get_args
from utils.utils import init_seeds, init_wandb
from utils.metrics import eval_cifar
import torch
import os
import json
from shutil import rmtree
import wandb
from main import get_loss
from data.cifar_contrastive_loader import ContrastiveCifar, make_lossy_dataloader, ContrastiveCifarHard, ContrastiveCifarHardTrain
from torch.optim.lr_scheduler import StepLR


def train_loop(args, gen, enc, loss, gen_val):
    # train
    enc.train()
    loss.train()
    running_loss = 0
    best_rcorr_corrupt = -2
    best_rcorr_entropy = -2
    n_total_batches = max(args.n_phases, 1) * 2 * args.n_batches_per_half_phase
    for b in tqdm(range(n_total_batches), position=0, leave=True):
        if args.n_phases == 0:
            # Train parameters jointly
            # To avoid restarting the optimizer, do this only in the beginning
            if b == 0:
                params = list(enc.parameters())
                if args.l_learnable_params:
                    params += list(loss.parameters())
                optim = torch.optim.Adam(params, lr=args.lr)
                scheduler = StepLR(optim, step_size=args.n_batches_per_half_phase, gamma=args.lr_decrease_after_phase)
                # Use repeated mu when training kappa
                n_neg = args.n_neg
                training_phase = "joint"
        else:
            # Choose which parameter training phase we are in
            if b % args.n_batches_per_half_phase == 0:
                lr = args.lr * args.lr_decrease_after_phase**(b / args.n_batches_per_half_phase)
                if (b / args.n_batches_per_half_phase) % 2 == 0:
                    # time to train mu:
                    params = list(enc.parameters())
                    if args.l_learnable_params:
                        params += list(loss.parameters())
                    optim = torch.optim.Adam(params, lr=lr)
                    scheduler = StepLR(optim, step_size=args.n_batches_per_half_phase, gamma=args.lr_decrease_after_phase)
                    # Use single n when training mu
                    n_neg = 0
                    training_phase = "mu"
                else:
                    # time to train kappa
                    params = list(enc.parameters())
                    if args.l_learnable_params:
                        params += list(loss.parameters())
                    optim = torch.optim.Adam(params, lr=lr)
                    scheduler = StepLR(optim, step_size=args.n_batches_per_half_phase, gamma=args.lr_decrease_after_phase)
                    # Use repeated mu when training kappa
                    n_neg = args.n_neg
                    training_phase = "kappa"

        # Train
        optim.zero_grad()
        x_ref, x_pos, x_neg = gen.sample(n=args.bs, n_neg=n_neg)
        mu_ref, kappa_ref = enc(x_ref)
        # Need to do some reshaping since we have two batch size dimensions (batch and n_pos), but enc expects one
        pos_bs = x_pos.shape[:2]
        x_pos = torch.reshape(x_pos, [pos_bs[0] * pos_bs[1], *x_pos.shape[2:]])
        mu_pos, kappa_pos = enc(x_pos)
        mu_pos = torch.reshape(mu_pos, [*pos_bs, *mu_pos.shape[1:]])
        kappa_pos = torch.reshape(kappa_pos, [*pos_bs, *kappa_pos.shape[1:]])
        if n_neg > 0:
            # Need to do some reshaping since we have two batch size dimensions (batch and n_neg), but enc expects one
            neg_bs = x_neg.shape[:2]
            x_neg = torch.reshape(x_neg, [neg_bs[0] * neg_bs[1], *x_neg.shape[2:]])
            mu_neg, kappa_neg = enc(x_neg)
            mu_neg = torch.reshape(mu_neg, [*neg_bs, *mu_neg.shape[1:]])
            kappa_neg = torch.reshape(kappa_neg, [*neg_bs, *kappa_neg.shape[1:]])
        else:
            mu_neg = None
            kappa_neg = None
        # mu and kappa are not stored as individual networks, but as direction and norm of the same parameter
        # Hence, we need to turn off their gradients here instead of in the optimizers
        if training_phase == "mu":
            kappa_ref = kappa_ref.detach()
            kappa_pos = kappa_pos.detach()
            if n_neg > 0:
                kappa_neg = kappa_neg.detach()
        elif training_phase == "kappa":
            pass
        elif training_phase == "joint":
            pass
        cur_loss = loss(mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg)
        cur_loss.backward()
        running_loss += cur_loss.detach().cpu().item()
        optim.step()
        scheduler.step()

        # Val
        if b == 0 or (b+1) % args.eval_every == 0:
            if b == 0:
                avg_loss = running_loss
            else:
                avg_loss = running_loss / args.eval_every
            print(f'Loss: {avg_loss}')
            enc.eval()
            r1, mapr, rcorr_entropy, r1_corrupt, mapr_corrupt, rcorr_corrupt = \
                eval_cifar(args, enc, gen_val.get_dataloader(), make_lossy_dataloader(gen_val.data))
            results_dict = {"r1": r1,
                            "mapr": mapr,
                            "rcorr_entropy": rcorr_entropy,
                            "r1_corrupt": r1_corrupt,
                            "mapr_corrupt": mapr_corrupt,
                            "rcorr_corrupt": rcorr_corrupt,
                            "temperature": loss.kappa.detach().cpu().item(),
                            "loss": avg_loss,
                            "batches": b}
            if args.use_wandb:
                wandb.log(results_dict)
            with open(f'results/{args.savefolder}/results_val_after_{(b+1):06d}_batches.json', 'w') as f2:
                json.dump(results_dict, f2)

            # Save the model if it achieved a new best
            if rcorr_corrupt > best_rcorr_corrupt:
                best_rcorr_corrupt = rcorr_corrupt
                best_rcorr_entropy = rcorr_entropy
                torch.save(enc.state_dict(), f"results/{args.savefolder}/encoder_params.pth")

            enc.train()
            running_loss = 0

    enc.eval()
    print(f"Validation score in best epoch: rcorr_entropy: {best_rcorr_entropy}, rcorr_corrupt: {best_rcorr_corrupt}")

    return enc


def get_traindata(args):
    if args.traindata == "test_softlabels":
        gen = ContrastiveCifar(mode="train", seed=args.seed, batch_size=args.bs)
    elif args.traindata == "test_hardlabels":
        gen = ContrastiveCifarHard(mode="train", seed=args.seed, batch_size=args.bs)
    elif args.traindata == "train_hardlabels":
        gen = ContrastiveCifarHardTrain(mode="train", batch_size=args.bs)
    else:
        raise NotImplementedError("traindata is not implemented.")

    return gen


if __name__ == "__main__":
    args = get_args()

    ################### SETUP ###################
    init_seeds(args.seed)
    loss = get_loss(args)
    gen = get_traindata(args)
    gen_val = ContrastiveCifar(mode="val", seed=args.seed, batch_size=args.bs)
    enc = ResnetProbEncoder(dim_z=args.e_dim_z, post_kappa_min=args.e_post_kappa_min, post_kappa_max=args.e_post_kappa_max)

    # Clean up the output folder
    if args.train:
        os.makedirs("results", exist_ok=True)
        # Delete possible old results if we are (re-)training
        rmtree(f'results/{args.savefolder}', ignore_errors=True)
        os.makedirs(f'results/{args.savefolder}', exist_ok=True)
        with open(f'results/{args.savefolder}/parameters.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    init_wandb(args)

    ################### TRAIN ###################
    if args.train:
        train_loop(args, gen, enc, loss, gen_val)

    ################### TEST ###################
    if args.test:
        # Load best model
        enc.load_state_dict(torch.load(f"results/{args.savefolder}/encoder_params.pth"))

        gen_test = ContrastiveCifar(mode="test", seed=args.seed, batch_size=args.bs)
        r1, mapr, rcorr_entropy, r1_corrupt, mapr_corrupt, rcorr_corrupt = \
            eval_cifar(args, enc, gen_test.get_dataloader(), make_lossy_dataloader(gen_test.data),  "testset", True)
        results_dict = {"r1": r1,
                        "mapr": mapr,
                        "rcorr_entropy": rcorr_entropy,
                        "r1_corrupt": r1_corrupt,
                        "mapr_corrupt": mapr_corrupt,
                        "rcorr_corrupt": rcorr_corrupt}
        if args.use_wandb:
            wandb.log(results_dict)
        with open(f'results/{args.savefolder}/results_testset.json', 'w') as f2:
            json.dump(results_dict, f2)

    print("Fin.")
