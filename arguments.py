import argparse

def str2bool(v):
    """
    Thank to stackoverflow user: Maxim
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    :param v: A command line argument with values [yes, true, t, y, 1, True, no, false, f, n, 0, False]
    :return: Boolean version of the command line argument
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#######################################
def get_args():
    parser = argparse.ArgumentParser()

    ##### Generator Parameters
    parser.add_argument('--g_dim_z', default=10, type=int, help='Dimensionality of latent space.')
    parser.add_argument('--g_dim_x', default=10, type=int, help='Dimensionality of x space.')
    parser.add_argument('--g_dim_hidden', default=10, type=int, help='Dimensionality of hidden layers.')
    parser.add_argument('--g_n_hidden', default=1, type=int, help='Number of hidden layers. (The network will additionally have one dim_x->dim_hidden and one dim_hidden->dim_z layer)')
    parser.add_argument('--g_pos_kappa', default=20, type=int, help='kappa of the implicit positive distribution in the z space.')
    # For controlled experiment:
    parser.add_argument('--g_post_family', default="vmf", type=str, help="How GT posteriors should be distributed (vmf, Gaussian, Laplace). Note: Predicted posteriors are *always* vmf")
    parser.add_argument('--g_post_kappa_min', default=16, type=float, help='How concentrated GT posteriors should be at least. (Use float("Inf") for the crisp case)')
    parser.add_argument('--g_post_kappa_max', default=32, type=float, help='How concentrated GT posteriors should be at most. (Use float("Inf") for the crisp case)')
    parser.add_argument('--g_min_spread', default=0.5, type=float, help='How much area of the sphere the generator needs to span to be accepted (measured by maximum cosine distance between means). Default 1 accepts any generator.')

    ##### Encoder Parameters
    parser.add_argument('--e_dim_z', default=10, type=int, help='Dimensionality of latent space.')
    parser.add_argument('--e_dim_hidden', default=0, type=int, help='Dimensionality of hidden layers. Use 0 to use the standard Zimmermann setting (10*e_dim_z for first and last, 50*e_dim_z for others)')
    parser.add_argument('--e_n_hidden', default=6, type=int, help='Number of hidden layers.')
    parser.add_argument('--e_post_kappa_min', default=16, type=float, help='To which range the encoder posteriors kappas should be initialized. (Use float("Inf") for the crisp case)')
    parser.add_argument('--e_post_kappa_max', default=32, type=float, help='To which range the encoder posteriors kappas should be initialized. (Use float("Inf") for the crisp case)')

    ##### Training parameters
    parser.add_argument('--train', default=True, type=str2bool, help="Whether to train. If false, uses an already trained checkpoint.")
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate for network parameters.')
    parser.add_argument('--n_phases', default=1, type=int, help='If 0, train kappa and mu together for 2 * n_batces_per_half_phase batches. Otherwise, do n_phases of split training, each training first mu and then kappa.')
    parser.add_argument('--n_batches_per_half_phase', default=50000, type=int, help='Number of training epochs per half phase (i.e. per mu and per kappa)')
    parser.add_argument('--lr_decrease_after_phase', default=0.1, type=float, help="Factor to multiply lr by after each half phase.")
    parser.add_argument('--bs', default=128, type=int, help='Mini-Batchsize to use.')
    parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--n_neg', default=16, type=int, help='Number of negative samples per image. If 0, use rolled batch as negative samples.')
    parser.add_argument('--oversampling_factor', default=10, type=int, help="How many candidates to generate per wanted rejection sample (higher=faster, but more RAM)")
    # For CIFAR experiment
    parser.add_argument('--traindata', default="test_softlabels", type=str, help="Which training data to use. train_hardlabels uses the normal CIFAR-10 train set. test_hardlabels a subset of the CIFAR-10 test set. test_softlabels a subset of the CIFAR-10H soft label test set.")

    ##### Loss parameters
    parser.add_argument('--loss', default="MCInfoNCE", type=str, help="Which loss (MCInfoNCE, ELK, HedgedInstance)")
    parser.add_argument('--l_n_samples', default=128, type=int, help='Number of MC samples to calculate the loss.')
    parser.add_argument('--l_learnable_params', default=False, type=str2bool, help="Whether loss parameters (e.g., temperature) should be learnable.")
    parser.add_argument('--l_hib_a', default=1, type=float, help="The multiplication constant in the HIB loss.")
    parser.add_argument('--l_hib_b', default=0, type=float, help="The addition constant in the HIB loss.")

    ##### Test parameters
    parser.add_argument('--eval_every', default=500, type=int, help="After how many batches to evaluate during training.")
    parser.add_argument('--n_numerical_eval', default=10000, type=int, help="On how many x-samples should posteriors be numerically evaluated.")
    parser.add_argument('--n_graphical_eval', default=400, type=int, help="On how many x-samples should posteriors be graphically evaluated. (this might take a long time for high numbers) 0 to turn off.")
    parser.add_argument('--savefolder', default="test", type=str, help="Where to save the results")
    # for CIFAR experiment
    parser.add_argument('--test', default=True, type=str2bool, help="Whether to eval on the test set.")

    ##### Weights and Biases parameters
    parser.add_argument('--use_wandb', default=False, type=str2bool, help="Turns on or off wandb tracking")
    parser.add_argument('--wandb_key', default="ADD YOUR WANDB API KEY HERE", type=str, help="Your Wandb API key")
    parser.add_argument('--wandb_project', default="prob_contr_learning", type=str, help="Project name")

    return parser.parse_args()