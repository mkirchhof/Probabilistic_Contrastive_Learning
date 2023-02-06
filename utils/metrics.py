import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.vmf_sampler import VonMisesFisher
from utils.utils import pairwise_cos_sims, pairwise_l2_dists
from scipy.stats import spearmanr
from abc import abstractmethod, ABC
from collections import OrderedDict
import faiss
from mpl_toolkits.axes_grid1 import ImageGrid
from tueplots import bundles
bundles.icml2022(family="sans-serif", usetex=False, column="half", nrows=1)
plt.rcParams.update(bundles.icml2022())


def vis_2d_sphere(point_list, ax=None):
    # Input:
    #  point_list: a list of 2D numpy arrays. Each array will be plotted with its own color

    if ax is None:
        ax = plt.gca()

    for points in point_list:
        ax.scatter(points[:, 0], points[:, 1])

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')


def eval_generator_smoothness(gen, ax=None):

    if ax is None:
        ax = plt.gca()

    n = 100
    x = gen._sample_x(n)
    mu, _ = gen(x)
    x = x.detach().cpu()
    mu = mu.detach().cpu()

    cos_dist = (1 - pairwise_cos_sims(mu)) / 2
    l2_dist = pairwise_l2_dists(x)
    corr = np.corrcoef(cos_dist.numpy(), l2_dist.numpy())[1, 0]

    ax.scatter(l2_dist, cos_dist)
    ax.set_xlabel("l2 Distance in input space")
    ax.set_ylabel("cos dist in embedding space")
    ax.set_title(f'corr={corr}')


def numerical_eval(args, gen, enc, x_eval=None, print_results=True):
    def mse(a, b):
        return torch.mean(torch.sqrt((a - b)**2))

    with torch.no_grad():
        if x_eval is None:
            x_eval = gen._sample_x(args.n_eval)
        mu_enc, kappa_enc = enc(x_eval)
        mu_gen, kappa_gen = gen(x_eval)

    # Evaluate means.
    # We want them to be equal up to a rotation of the sphere.
    # Zimmermann do this by regressing the mu_enc and mu_gen, saying they should be r=1 if one is only a rotation
    # of the other. Not sure if this holds. Rotation matrices can have imaginary parts.
    # We will do it by comparing the pairwise cosine distances between the mu_s
    cosdist_enc = pairwise_cos_sims(mu_enc)
    cosdist_gen = pairwise_cos_sims(mu_gen)
    cosdist_mse = mse(cosdist_gen, cosdist_enc)
    cosdist_corr = np.corrcoef(cosdist_gen.cpu().numpy(), cosdist_enc.cpu().numpy())[1, 0]
    cosdist_rankcorr = spearmanr(cosdist_gen.cpu().numpy(), cosdist_enc.cpu().numpy())[0]

    # Evaluate kappas
    # For kappas, we want exact match. But possibly if we are matching up to a scale, that might indicate something.
    kappa_mse = mse(kappa_enc, kappa_gen)
    kappa_corr = np.corrcoef(kappa_enc.cpu().flatten().numpy(), kappa_gen.cpu().flatten().numpy())[1, 0]
    kappa_rankcorr = spearmanr(kappa_enc.cpu().flatten().numpy(), kappa_gen.cpu().flatten().numpy())[0]

    if print_results:
        print(f'Cosdist MSE = {cosdist_mse:.3f}, corr = {cosdist_corr:.3f}, rankcorr = {cosdist_rankcorr:.3f}. Kappa MSE = {kappa_mse:.3f}, corr = {kappa_corr:.3f}, rankcorr = {kappa_rankcorr:.3f}.')
    return cosdist_mse, cosdist_corr, cosdist_rankcorr, kappa_mse, kappa_corr, kappa_rankcorr


def graphical_eval(args, gen, enc, x_eval=None, print_examples=False):
    with torch.no_grad():
        if x_eval is None:
            x_eval = gen._sample_x(args.n_eval)
        mu_enc, kappa_enc = enc(x_eval)
        mu_gen, kappa_gen = gen(x_eval)
        n_examples = torch.arange(0, x_eval.shape[0] - 1, torch.floor(torch.ones(1) * (x_eval.shape[0] - 1) / 30).type(torch.long).item())
        examples_ids = torch.argsort(kappa_gen.squeeze())[n_examples]
        samples_vmf_enc = VonMisesFisher(mu_enc[examples_ids], kappa_enc[examples_ids]).sample(10)
        samples_vmf_gen = VonMisesFisher(mu_gen[examples_ids], kappa_gen[examples_ids]).sample(10)

    # The 2D sphere of enc and gen should be the same, up to a rotation
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9.5, 9.5))
    vis_2d_sphere([samples_vmf_gen[:, i, :].cpu().numpy() for i in range(samples_vmf_gen.shape[1])], ax1)
    vis_2d_sphere([samples_vmf_enc[:, i, :].cpu().numpy() for i in range(samples_vmf_enc.shape[1])], ax2)
    ax1.set_title("Generator space $\mathcal{Z}$")
    ax2.set_title("Encoder space $\mathcal{E}$")

    # Show the correlation between the cosine dists
    cosdist_enc = pairwise_cos_sims(mu_enc)
    cosdist_gen = pairwise_cos_sims(mu_gen)
    ax3.scatter(cosdist_gen.cpu().numpy(), cosdist_enc.cpu().numpy())
    ax3.set_xlabel("Generator pairwise cos-sim")
    ax3.set_ylabel("Encoder pairwise cos-sim")

    # Show the correlation between the kappas
    ax4.scatter(kappa_gen.cpu().flatten().numpy(), kappa_enc.cpu().flatten().numpy())
    ax4.set_xlabel("Generator $\kappa$")
    ax4.set_ylabel("Encoder $\kappa$")

    # Print some exemplary predictions
    if print_examples:
        print("Exemplary true and predicted mu and kappas:")
        print("GT mu(x):")
        print(mu_gen[examples_ids])
        print("GT kappa(x):")
        print(kappa_gen[examples_ids])
        print("Pred mu(x):")
        print(mu_enc[examples_ids])
        print("Pred kappa(x):")
        print(kappa_enc[examples_ids])

def eval_cifar(args, enc, dataloader, corrupted_dataloader, filename_suffix=None, want_plot=False):
    with torch.no_grad():
        # Run through the dataset:
        embeddings = list()
        confidences = list()
        soft_labels = list()
        for img, soft_label in dataloader:
            embed, conf = enc(img.to(enc.device))
            embeddings.append(embed.detach().cpu())
            confidences.append(conf.detach().cpu().squeeze())
            soft_labels.append(soft_label)
        embeddings = torch.cat(embeddings, dim=0)
        confidences = torch.cat(confidences, dim=0)
        soft_labels = torch.cat(soft_labels, dim=0)
        _, hard_labels = torch.max(soft_labels, dim=1)
        gt_entropy = soft_labels * torch.log(soft_labels)
        gt_entropy[torch.isnan(gt_entropy)] = 0
        gt_entropy = torch.sum(gt_entropy, dim=1)

        # predicted entropy vs class entropy
        rcorr_entropy = spearmanr(confidences.numpy(), gt_entropy.numpy())[0]

        # MAP@R vs confidence filter-out rates
        mapr = NearestNeighboursMetrics()(embeddings, hard_labels, confidences)

        # predicted entropy vs cifar10s entropy
        # rcorr_cifar10s = spearmanr(confidences.numpy()[np.logical_not(np.isnan(entropies_cifar10s))],
        #                            entropies_cifar10s[np.logical_not(np.isnan(entropies_cifar10s))])[0]
        # print(rcorr_cifar10s)

        # Plots
        if want_plot:
            if filename_suffix is not None:
                path = f"results/{args.savefolder}/uncertain_images_{filename_suffix}.png"
                path_erc = f"results/{args.savefolder}/erc_plot_{filename_suffix}.pdf"
                path_conf_vs_entropy = f"results/{args.savefolder}/confidence_vs_entropy_{filename_suffix}.png"
                path_conf_vs_corruption = f"results/{args.savefolder}/confidence_vs_corruption_{filename_suffix}.png"
                path_uncertain_retrieval = f"results/{args.savefolder}/uncertain_retrieval_{filename_suffix}.png"
                # path_conf_vs_cifar10s = f"results/{args.savefolder}/confidence_vs_cifar10s_{filename_suffix}.png"
            else:
                path = f"results/{args.savefolder}/uncertain_images.png"
                path_erc = f"results/{args.savefolder}/erc_plot.pdf"
                path_conf_vs_entropy = f"results/{args.savefolder}/confidence_vs_entropy.png"
                path_conf_vs_corruption = f"results/{args.savefolder}/confidence_vs_corruption.png"
                path_uncertain_retrieval = f"results/{args.savefolder}/uncertain_retrieval.png"
                # path_conf_vs_cifar10s = f"results/{args.savefolder}/confidence_vs_cifar10s.png"
            # path_cifar_10h_vs_10s = "cifar_10h_vs_10s.png"
            #query_ids = torch.multinomial(torch.ones(embeddings.shape[0]), 5, replacement=False)
            with plt.rc_context(bundles.icml2022()):
                erc_plot(mapr["erc-recall@1"], path_erc)
            scatter_plot(confidences.numpy(), gt_entropy.numpy(), path_conf_vs_entropy,
                         xlabel="Predicted Confidence $\kappa$", ylabel="Negative Entropy of CIFAR-10H")
            query_ids = torch.argsort(confidences, descending=False)[torch.Tensor([30, 500, 1000, 1302, 1800]).type(torch.long)]
            print("Kappa values of the example images:")
            print(confidences[query_ids])
            #query_ids = torch.arange(0, embeddings.shape[0] - 1, torch.floor(torch.ones(1) * embeddings.shape[0]).item() / 5).type(torch.long)
            uncertain_retrieval(embeddings[query_ids], confidences[query_ids], query_ids,
                                embeddings, confidences, dataloader.dataset, path=path_uncertain_retrieval)
            uncertain_images(confidences, hard_labels, dataloader, path)
            # scatter_plot(confidences.numpy()[np.logical_not(np.isnan(entropies_cifar10s))],
            #              entropies_cifar10s[np.logical_not(np.isnan(entropies_cifar10s))], path_conf_vs_cifar10s,
            #              xlabel="Predicted Confidence $\kappa$", ylabel="Negative Entropy of CIFAR-10S")
            # scatter_plot(gt_entropy.numpy()[np.logical_not(np.isnan(entropies_cifar10s))],
            #              entropies_cifar10s[np.logical_not(np.isnan(entropies_cifar10s))], path_cifar_10h_vs_10s,
            #              xlabel="Negative Entropy of CIFAR-10H", ylabel="Negative Entropy of CIFAR-10S")

        # Run through the corrupted data
        embeddings = list()
        confidences = list()
        soft_labels = list()
        quality = list()
        for img, soft_label, qual in corrupted_dataloader:
            embed, conf = enc(img.to(enc.device))
            embeddings.append(embed.detach().cpu())
            confidences.append(conf.detach().cpu().squeeze())
            soft_labels.append(soft_label)
            quality.append(qual)
        embeddings = torch.cat(embeddings, dim=0)
        confidences = torch.cat(confidences, dim=0)
        soft_labels = torch.cat(soft_labels, dim=0)
        _, hard_labels = torch.max(soft_labels, dim=1)
        quality = torch.cat(quality, dim=0)

        # predicted entropy vs corruption level
        rcorr_corrupt = spearmanr(confidences.numpy(), quality.numpy())[0]

        # MAP@R vs confidence filter-out rates on corrupted data
        mapr_corrupt = NearestNeighboursMetrics()(embeddings, hard_labels, confidences)

        # Another plot
        if want_plot:
            scatter_plot(confidences.numpy(), quality.numpy(), path_conf_vs_corruption,
                         xlabel="Predicted Confidence $\kappa$", ylabel="Proportion of Image Shown")

    return mapr["recall@1"], mapr["mapr"].detach().cpu().item(), rcorr_entropy, \
        mapr_corrupt["recall@1"], mapr_corrupt["mapr"].detach().cpu().item(), rcorr_corrupt

def erc_plot(erc, path="erc_plot.png"):
    # erc - a tensor with the cumsums of average errors for the least confident 1, 2, 3, 4, ... samples
    plt.figure(figsize=(3.25, 2.))
    plt.plot(np.arange(1, len(erc) + 1) / len(erc), erc, color="#4878d0")
    plt.xlabel("Percentage of Excluded Lowest-certain Samples")
    plt.ylabel("Recall@1")
    plt.grid(zorder=-1, color="lightgrey", lw=0.5)
    plt.savefig(path)
    plt.close()

    return None

def scatter_plot(x, y, path="scatterplot.png", xlabel="", ylabel=""):
    # erc - a tensor with the cumsums of average errors for the least confident 1, 2, 3, 4, ... samples
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()

    return None


def plot_images(dataset, ids, fig=plt.figure(figsize=(20., 20.))):
    # Provided a 2D tensor of image ids, picks them from the dataset and plots them in the
    # same matrix structure they have in ids
    # If ids includes negative ids, they are skipped and plotted as a white picture

    # Setup image grid
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(ids.shape[0], ids.shape[1]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    # Plot images
    # Iterating over the grid returns the Axes.
    for ax, id in zip(grid, ids.flatten()):
        ax.set_axis_off()
        if id >= 0:
            im, _ = dataset.__getitem__(id)
            ax.imshow(torch.minimum(torch.ones(1), torch.maximum(torch.zeros(1), im.permute(1, 2, 0) *
                                                                 torch.tensor([0.2471, 0.2435, 0.2616]) + torch.tensor(
                [0.4914, 0.4822, 0.4465]))))


def uncertain_images(confidence, labels, dataloader, path="uncertain_images.png"):
    # Find most certain/uncertain images per class
    chosenclasses = np.array(np.arange(10))
    chosen_ids = []
    for lab in chosenclasses:
        ids = np.array([i for i in np.arange(len(labels)) if labels[i] == lab])
        order = torch.argsort(confidence[ids])
        first = ids[order[:5]]
        last = ids[order[-5:]]
        chosen_ids.append(np.concatenate((first, last)).tolist())
    chosen_ids = np.array(chosen_ids)
    chosen_ids[1, 1] = -1

    fig = plt.figure(figsize=(20., 20.))
    plot_images(dataloader.dataset, chosen_ids, fig)
    fig.savefig(path)
    plt.close()

    return None

def uncertain_retrieval(q_embed, q_conf, q_id, r_embed, r_conf, dataset, alpha=0.05, path="uncertain_retrieval.png"):
    '''
    Retrieves and plots the top images for each query image from a retrieval dataset
    :param q_embed: tensor of shape [n_query, dim], mean embeds of images to be searched
    :param q_conf: tensor of shape [n_query], kappa confidence values of images to be searched
    :param r_embed: tensor of shape [retrieval_dataset_size, dim], mean embeds of all images in the desired dataset
    :param r_conf: tensor of shape [retrieval_dataset_size], kappa confidence values of all images in the desired dataset
                    (currently unused)
    :param dataset: a dataset object where we can retrieve images via a __getitem__ method
    :param alpha: The "confidence level" for the maximum-a-posteriori interval
    :param path: String, where to save the image
    :return: Nothing, but a plot is plotted
    '''
    # order queries by their confidence
    order = torch.argsort(q_conf, descending=True)
    q_embed = q_embed[order]
    q_id = q_id[order]
    q_conf = q_conf[order]
    q_conf = q_conf.unsqueeze(1)

    # Calculate confidence intervals
    samples = VonMisesFisher(q_embed, q_conf).sample(10000)
    dot_prods = torch.sum(q_embed.unsqueeze(0) * samples, dim=-1)
    thresholds = torch.quantile(dot_prods, alpha, dim=0)

    # See which region of the embedding space this covers to decide how many samples we should show
    unif_samples = torch.zeros((1000000, r_embed.shape[1])).normal_()
    unif_samples = unif_samples / unif_samples.norm(dim=1).unsqueeze(1)
    dot_prods = torch.sum(q_embed[0].unsqueeze(0) * unif_samples, dim=-1)
    covered_sphere_pct = torch.sum(dot_prods.unsqueeze(0) >= thresholds.unsqueeze(1), dim=1) / 1000000
    n_select = torch.round(covered_sphere_pct / torch.max(covered_sphere_pct) * 21).type(torch.long) # we want 14 for the maximum one

    # collect query image ids that fall into these confidence intervals
    dot_prod_q_r = torch.sum(q_embed.unsqueeze(1) * r_embed.unsqueeze(0), dim=-1)
    is_in_interval = dot_prod_q_r >= thresholds.unsqueeze(1)
    # Prevent retrieving the query image itself
    for i, id in enumerate(q_id):
        is_in_interval[i, id] = 0
    # Edge case: No similarities found. Then just remove the thing
    n_found = is_in_interval.sum(dim=1)
    q_id = q_id[n_found > 0]
    is_in_interval = is_in_interval[n_found > 0,:]
    # The space is very clustered. Incentivize to pick not all images from the same cluster by
    # weighing the samples with the inverse density of the embedding space at its position, estimated by NN
    #dens = torch.sum(torch.exp(80 * torch.sum(r_embed.unsqueeze(0) * r_embed.unsqueeze(1), dim=-1)), dim=1)
    selected_ids = torch.multinomial(is_in_interval.type(torch.float), torch.max(n_select), replacement=False)# / dens.unsqueeze(0), torch.max(n_select), replacement=False)
    # Remember to only use the the n_select first ones of each row.
    # We only had to choose the max for all, because a tensor cannot have different number of entries in each row.

    # plot
    # Transform the selected_ids into an array that will look good in uncertain_images
    ncol = 7 # Columns of retrieved images
    dist_between_query_and_retrieval = 4
    dist_between_retrieval_rows = 6
    retrieval_tensors = []
    for i in range(selected_ids.shape[0]):
        nrows = ((n_select[i] - 1) // (ncol) + 1) + dist_between_retrieval_rows
        id_array = -torch.ones(nrows * (ncol))
        selected = selected_ids[i, :n_select[i]]
        # Order them by similarity
        similarities = dot_prod_q_r[i, selected]
        selected = selected[torch.argsort(similarities, descending=True)]
        id_array[:n_select[i]] = selected
        id_array = torch.reshape(id_array, (nrows, ncol))
        query_array = -torch.ones((nrows, dist_between_query_and_retrieval))
        query_array[0, 0] = q_id[i]
        id_array = torch.cat((query_array, id_array), dim=1)
        retrieval_tensors.append(id_array)
    id_matrix = torch.cat(retrieval_tensors, dim=0).type(torch.long).numpy()

    fig = plt.figure(figsize=(20., 20.))
    plot_images(dataset, id_matrix, fig)
    fig.savefig(path)
    plt.close()

########################################################################################################################
# The following code is modified from https://github.com/tinkoff-ai/probabilistic-embeddings, Apache License 2.0       #
########################################################################################################################
def asarray(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    return np.ascontiguousarray(x)


class NearestNeighboursBase(ABC):
    """Base class for all nearest neighbour metrics."""

    @property
    @abstractmethod
    def match_self(self):
        """Whether to compare each sample with self or not."""
        pass

    @property
    @abstractmethod
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        pass

    @property
    @abstractmethod
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        pass

    @abstractmethod
    def num_nearest(self, labels):
        """Get the number of required neighbours.
        Args:
            labels: Dataset labels.
        """
        pass

    @abstractmethod
    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores=None, confidences=None):
        """Compute metric value.
        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Class size for each element.
            positive_scores (optional): Similarity scores of elements with the same class (depends on match_self).
            confidences (optional): Confidence for each element of the batch with shape (B).
        Returns:
            Metric value.
        """
        pass


class RecallK(NearestNeighboursBase):
    """Recall@K metric."""
    def __init__(self, k):
        self._k = k

    @property
    def match_self(self):
        """Whether to compare each sample with self or not."""
        return False

    @property
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        return False

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return False

    def num_nearest(self, labels):
        """Get the number of required neighbours.
        Args:
            labels: Dataset labels.
        """
        return self._k

    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores=None, confidences=None):
        """Compute metric value.
        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Class size for each element.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        Returns:
            Metric value.
        """
        mask = class_sizes > 1
        if mask.sum().item() == 0:
            return np.nan
        has_same, _ = nearest_same[mask, :self._k].max(1)
        return has_same.float().mean().item()


class ERCRecallK(NearestNeighboursBase):
    """Error-versus-Reject-Curve based on Recall@K metric."""
    def __init__(self, k):
        self._k = k

    @property
    def match_self(self):
        """Whether to compare each sample with self or not."""
        return False

    @property
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        return False

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return True

    def num_nearest(self, labels):
        """Get the number of required neighbours.
        Args:
            labels: Dataset labels.
        """
        return self._k

    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores=None, confidences=None):
        """Compute metric value.
        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Class size for each element.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        Returns:
            Metric value.
        """
        if confidences is None:
            raise ValueError("Can't compute ERC without confidences.")
        mask = class_sizes > 1
        if mask.sum().item() == 0:
            return np.nan
        recalls, _ = nearest_same[mask, :self._k].max(1)
        errors = 1 - recalls.float()
        confidences = confidences[mask]

        b = len(errors)
        order = torch.argsort(confidences, descending=True)
        errors = errors[order]  # High confidence first.
        mean_errors = errors.cumsum(0) / torch.arange(1, b + 1, device=errors.device)
        # We want to plot the R@1, not the error, so return the correct predictions
        correct = 1 - mean_errors
        correct = torch.flip(correct, dims=(0, )) # In the latter plot, we want the highest confident samples last
        return correct.cpu().numpy()

class ATRBase(NearestNeighboursBase):
    """Base class for @R metrics.
    All @R metrics search for the number of neighbours equal to class size.
    Args:
        match_self: Whether to compare each sample with self or not.
    Inputs:
        - parameters: Embeddings distributions tensor with shape (B, P).
        - labels: Label for each embedding with shape (B).
    Outputs:
        - Metric value.
    """

    def __init__(self, match_self=False):
        super().__init__()
        self._match_self = match_self

    @property
    @abstractmethod
    def oversample(self):
        """Sample times more nearest neighbours."""
        pass

    @abstractmethod
    def _aggregate(self, nearest_same, nearest_scores, num_nearest, class_sizes, positive_scores, confidences=None):
        """Compute metric value.
        Args:
            nearest_same: Matching labels for nearest neighbours with shape (B, R).
                Matches are coded with 1 and mismatches with 0.
            nearest_scores: Score for each neighbour with shape (B, R).
            num_nearest: Number of nearest neighbours for each element of the batch with shape (B).
            class_sizes: Number of elements in the class for each element of the batch.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        """
        pass

    @property
    def match_self(self):
        """Whether to compare each sample with self or not."""
        return self._match_self

    @property
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        return True

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return False

    def num_nearest(self, labels):
        """Get maximum number of required neighbours.
        Args:
            labels: Dataset labels.
        """
        max_r = torch.bincount(labels).max().item()
        max_r *= self.oversample
        return max_r

    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores, confidences=None):
        """Compute metric value.
        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Number of elements in the class for each element of the batch.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        Returns:
            Metric value.
        """
        num_positives = class_sizes if self.match_self else class_sizes - 1
        num_nearest = torch.clip(num_positives * self.oversample, max=nearest_same.shape[1])
        return self._aggregate(nearest_same, nearest_scores, num_nearest, class_sizes, positive_scores,
                               confidences=confidences)


class MAPR(ATRBase):
    """MAP@R metric.
    See "A Metric Learning Reality Check" (2020) for details.
    """

    @property
    def oversample(self):
        """Sample times more nearest neighbours."""
        return 1

    def _aggregate(self, nearest_same, nearest_scores, num_nearest, class_sizes, positive_scores, confidences=None):
        """Compute MAP@R.
        Args:
            nearest_same: Matching labels for nearest neighbours with shape (B, R).
                Matches are coded with 1 and mismatches with 0.
            nearest_scores: (unused) Score for each neighbour with shape (B, R).
            num_nearest: Number of nearest neighbours for each element of the batch with shape (B).
            class_sizes: (unused) Number of elements in the class for each element of the batch.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        """
        b, r = nearest_same.shape
        device = nearest_same.device
        range = torch.arange(1, r + 1, device=device)  # (R).
        count_mask = range[None].tile(b, 1) <= num_nearest[:, None]  # (B, R).
        precisions = count_mask * nearest_same * torch.cumsum(nearest_same, dim=1) / range[None]  # (B, R).
        maprs = precisions.sum(-1) / torch.clip(num_nearest, min=1)  # (B).
        return maprs.mean()


class ERCMAPR(ATRBase):
    """ERC curve for MAP@R metric."""

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return True

    @property
    def oversample(self):
        """Sample times more nearest neighbours."""
        return 1

    def _aggregate(self, nearest_same, nearest_scores, num_nearest, class_sizes, positive_scores, confidences=None):
        """Compute MAP@R ERC.
        Args:
            nearest_same: Matching labels for nearest neighbours with shape (B, R).
                Matches are coded with 1 and mismatches with 0.
            nearest_scores: (unused) Score for each neighbour with shape (B, R).
            num_nearest: Number of nearest neighbours for each element of the batch with shape (B).
            class_sizes: (unused) Number of elements in the class for each element of the batch.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        """
        if confidences is None:
            raise ValueError("Can't compute ERC without confidences.")
        b, r = nearest_same.shape
        device = nearest_same.device
        range = torch.arange(1, r + 1, device=device)  # (R).
        count_mask = range[None].tile(b, 1) <= num_nearest[:, None]  # (B, R).
        precisions = count_mask * nearest_same * torch.cumsum(nearest_same, dim=1) / range[None]  # (B, R).
        maprs = precisions.sum(-1) / torch.clip(num_nearest, min=1)  # (B).
        errors = 1 - maprs.float()

        b = len(errors)
        order = torch.argsort(confidences, descending=True)
        errors = errors[order]  # High confidence first.
        mean_errors = errors.cumsum(0) / torch.arange(1, b + 1, device=errors.device)
        return mean_errors.mean().cpu().item()


class KNNIndex:
    BACKENDS = {
        "faiss": faiss.IndexFlatL2
    }

    def __init__(self, dim, backend="torch"):
        self._index = self.BACKENDS[backend](dim)

    def __enter__(self):
        if self._index is None:
            raise RuntimeError("Can't create context multiple times.")
        return self._index

    def __exit__(self, exc_type, exc_value, traceback):
        self._index.reset()
        self._index = None

class NearestNeighboursMetrics:
    """Metrics based on nearest neighbours search.
    Args:
        distribution: Distribution object.
        scorer: Scorer object.
    Inputs:
        - parameters: Embeddings distributions tensor with shape (B, P).
        - labels: Label for each embedding with shape (B).
    Outputs:
        - Metrics values.
    """

    METRICS = {
        "recall": RecallK,
        "erc-recall@1": lambda: ERCRecallK(1),
        "mapr": MAPR,
        "erc-mapr": ERCMAPR
    }

    @staticmethod
    def get_default_config(backend="faiss", broadcast_backend="torch", metrics=None, prefetch_factor=2, recall_k_values=(1,5)):
        """Get metrics parameters.
        Args:
            backend: KNN search engine ("faiss").
            broadcast_backend: Torch doesn't support broadcast for gather method.
              We can emulate this behaviour with Numpy ("numpy") or tiling ("torch").
            metrics: List of metric names to compute ("recall", "mapr", "mapr-nms").
                By default compute all available metrics.
            prefetch_factor: Nearest neighbours number scaler for presampling.
            recall_k_values: List of K values to compute recall at.
        """
        return OrderedDict([
            ("backend", backend),
            ("broadcast_backend", broadcast_backend),
            ("metrics", metrics),
            ("prefetch_factor", prefetch_factor),
            ("recall_k_values", recall_k_values)
        ])

    def __init__(self, *, match_via_cosine=True):
        self._config = self.get_default_config()

        self._metrics = OrderedDict()
        metric_names = self._config["metrics"] if self._config["metrics"] is not None else list(self.METRICS)
        for name in metric_names:
            if name == "recall":
                for k in self._config["recall_k_values"]:
                    k = int(k)
                    self._metrics["{}@{}".format(name, k)] = self.METRICS[name](k)
            else:
                metric = self.METRICS[name]()
                self._metrics[name] = metric

        self.match_via_cosine = match_via_cosine

    def __call__(self, embeddings, labels, confidences=None):
        """
        This computes the metrics of given embeddings
        :param embeddings: (Batchsize, dim) tensor of embeddings
        :param labels: (Batchsize) tensor of class labels
        :param confidences: (Batchsize) tensor of confidences (higher = more confident).
                            If None, will use the distance to the closest neighbor
        :return:
        """
        if len(labels) != len(embeddings):
            raise ValueError("Batch size mismatch between labels and embeddings.")
        embeddings = embeddings.detach()  # (B, P).
        labels = labels.detach()  # (B).

        if self.match_via_cosine:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        # Find desired nearest neighbours number for each sample and total.
        label_counts = torch.bincount(labels)  # (L).
        class_sizes = label_counts[labels]  # (B).
        num_nearest = max(metric.num_nearest(labels) + int(not metric.match_self) for metric in self._metrics.values())
        num_nearest = min(num_nearest, len(labels))

        # Gather nearest neighbours (sorted in score descending order).
        nearest, scores = self._find_nearest(embeddings, num_nearest)  # (B, R), (B, R).
        num_nearest = torch.full((len(nearest),), num_nearest, device=labels.device)
        nearest_labels = self._gather_broadcast(labels[None], 1, nearest, backend=self._config["broadcast_backend"])  # (B, R).
        nearest_same = nearest_labels == labels[:, None]  # (B, R).

        # If necessary, compute confidence as the similarity to the closest neighbor
        need_confidences = any([metric.need_confidences for metric in self._metrics.values()])
        if need_confidences and confidences is None:
            confidences = scores[:,0]

        need_positives = any(metric.need_positives for metric in self._metrics.values())
        if need_positives:
            positive_scores, _, positive_same_mask = self._get_positives(embeddings, labels)
        else:
            positive_scores, positive_same_mask = None, None

        need_nms = any(not metric.match_self for metric in self._metrics.values())
        if need_nms:
            no_self_mask = torch.arange(len(labels), device=embeddings.device)[:, None] != nearest
            nearest_same_nms, _ = self._gather_mask(nearest_same, num_nearest, no_self_mask)
            scores_nms, num_nearest = self._gather_mask(scores, num_nearest, no_self_mask)
            if need_positives:
                positive_scores_nms, _ = self._gather_mask(positive_scores, class_sizes, ~positive_same_mask)
            else:
                positive_scores_nms = None

        metrics = OrderedDict()
        for name, metric in self._metrics.items():
            if metric.match_self:
                metrics[name] = metric(nearest_same, scores, class_sizes, positive_scores, confidences=confidences)
            else:
                metrics[name] = metric(nearest_same_nms, scores_nms, class_sizes, positive_scores_nms, confidences=confidences)
        return metrics

    def _find_nearest(self, embeddings, max_nearest):
        """Find nearest neighbours for each element of the batch.
        """
        embeddings = embeddings.unsqueeze(1) # We only have one "mode"
        b, c, d = embeddings.shape
        # Find neighbors using simple L2/dot scoring.
        prefetch = min(max_nearest * self._config["prefetch_factor"], b)
        candidates_indices, sim = self._multimodal_knn(embeddings, prefetch)  # (B, C * R).
        return candidates_indices.reshape((b, -1)), sim.reshape((b, -1))

    def _get_positives(self, embeddings, labels):
        label_counts = torch.bincount(labels)
        num_labels = len(label_counts)
        max_label_count = label_counts.max().item()
        by_label = torch.full((num_labels, max_label_count), -1, dtype=torch.long)
        counts = np.zeros(num_labels, dtype=np.int64)
        for i, label in enumerate(labels.cpu().numpy()):
            by_label[label][counts[label]] = i
            counts[label] += 1
        by_label = by_label.to(labels.device)  # (L, C).
        indices = by_label[labels]  # (B, C).
        num_positives = torch.from_numpy(counts).long().to(labels.device)[labels]
        positive_parameters = self._gather_broadcast(embeddings[None], 1, indices[..., None],
                                                     backend=self._config["broadcast_backend"])  # (B, C, P).
        with torch.no_grad():
            positive_scores = torch.sum(embeddings[:, None, :] *  positive_parameters, dim = -1)  # (B, C).
        same_mask = indices == torch.arange(len(labels), device=indices.device)[:, None]
        # Sort first elements in each row according to counts.
        no_sort_mask = torch.arange(positive_scores.shape[1], device=embeddings.device)[None] >= num_positives[:, None]
        positive_scores[no_sort_mask] = positive_scores.min() - 1
        positive_scores, order = torch.sort(positive_scores, dim=1, descending=True)
        same_mask = torch.gather(same_mask, 1, order)
        return positive_scores, num_positives, same_mask

    def _multimodal_knn(self, x, k):
        """Find nearest neighbours for multimodal queries.
        Args:
            x: Embeddings with shape (B, C, D) where C is the number of modalities.
            k: Number of nearest neighbours.
        Returns:
            Nearest neighbours indices with shape (B, C, K). Indices are in the range [0, B - 1].
        """
        b, c, d = x.shape
        if k > b:
            raise ValueError("Number of nearest neighbours is too large: {} for batch size {}.".format(k, b))
        x_flat = asarray(x).reshape((b * c, d))
        with KNNIndex(d, backend=self._config["backend"]) as index:
            index.add(x_flat)
            sim, indices = index.search(x_flat, k)  # (B * C, K), indices are in [0, B * C - 1].
        indices //= c  # (B * C, K), indices are in [0, B - 1].
        return torch.from_numpy(indices.reshape((b, c, k))).long().to(x.device), torch.from_numpy(sim.reshape((b, c, k))).to(x.device)

    @staticmethod
    def _remove_duplicates(indices, num_unique):
        """Take first n unique values from each row.
        Args:
            indices: Input indices with shape (B, K).
            num_unique: Number of unique indices in each row.
        Returns:
            Unique indices with shape (B, num_unique) and new scores if scores are provided.
        """
        b, k = indices.shape
        if k == 1:
            return indices
        sorted_indices, order = torch.sort(indices, dim=1, stable=True)
        mask = sorted_indices[:, 1:] != sorted_indices[:, :-1]  # (B, K - 1).
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)  # (B, K).
        mask = torch.gather(mask, 1, torch.argsort(order, dim=1))
        counts = torch.cumsum(mask, 1)  # (B, K).
        mask &= counts <= num_unique  # (B, K).

        # Some FAISS indices allow duplicates. In this case total number of unique elements is less than min_unique.
        # Add tail samples to get exact min_unique number.
        num_extra_zeros = torch.clip(num_unique - counts[:, -1], 0)
        counts = torch.cumsum(~mask, 1)
        sums = counts[:, -1].unsqueeze(-1)  # (B, 1).
        counts = torch.cat((sums, sums - counts[:, :-1]), dim=-1)  # (B, K).
        mask |= counts <= num_extra_zeros[:, None]

        unique = indices[mask].reshape(b, num_unique)  # (B, R), all indices are unique.
        return unique

    @staticmethod
    def _gather_mask(matrix, lengths, mask):
        b, n = matrix.shape
        device = matrix.device
        length_mask = torch.arange(n, device=device)[None].tile(b, 1) < lengths[:, None]  # (B, N).
        mask = mask & length_mask
        counts = mask.sum(1)  # (B).
        max_count = counts.max()
        padding = max_count - counts.min()
        if padding > 0:
            matrix = torch.cat((matrix, torch.zeros(b, padding, dtype=matrix.dtype, device=device)), dim=1)
            mask = torch.cat((mask, torch.ones(b, padding, dtype=torch.bool, device=device)), dim=1)
        mask &= torch.cumsum(mask, 1) <= max_count
        return matrix[mask].reshape(b, max_count), counts

    @staticmethod
    def _gather_broadcast(input, dim, index, backend="torch"):
        if backend == "torch":
            shape = np.maximum(np.array(input.shape), np.array(index.shape)).tolist()
            index[index < 0] += shape[dim]
            shape[dim] = input.shape[dim]
            input = input.broadcast_to(shape)
            shape[dim] = index.shape[dim]
            index = index.broadcast_to(shape)
            return input.gather(dim, index)
        elif backend == "numpy":
            result_array = np.take_along_axis(asarray(input),
                                              asarray(index),
                                              dim)
            result = torch.from_numpy(result_array).to(dtype=input.dtype, device=input.device)
            return result
        else:
            raise ValueError("Unknown broadcast backend: {}.".format(backend))