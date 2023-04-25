from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as functional_transforms
from PIL import Image
import os
import json

class ContrastiveCifar():
    def __init__(self, mode="train", seed=1, batch_size=64, device=torch.device("cuda:0")):
        super().__init__()
        self.device = device

        # Load data
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
        self.data = datasets.CIFAR10(root='data/data_CIFAR10_test', train=False, download=True, transform=self.transform)
        plabels_path = "data/cifar10h-probs.npy"
        if os.path.isfile(plabels_path):
            self.plabels = torch.from_numpy(np.load(plabels_path))
        else:
            raise FileNotFoundError("Could not find CIFAR-10H labels under " + plabels_path + ". Please download them (see README -> Installation).")

        # Limit to train/val/test
        # Each idx_set_i.csv contains 2000 image ids as crossvalidation splits of the original 10000 idxes.
        # We use i = {seed, seed + 1, seed + 2} MOD 5 for train,
        # i = seed + 3 MOD 5 for val
        # i = seed + 4 MOD 4 for test
        if mode == "train":
            train_1 = np.loadtxt(f"data/idx_set_{seed % 5}.csv", delimiter=",").astype("int")
            train_2 = np.loadtxt(f"data/idx_set_{(seed + 1) % 5}.csv", delimiter=",").astype("int")
            train_3 = np.loadtxt(f"data/idx_set_{(seed + 2) % 5}.csv", delimiter=",").astype("int")
            idxes = np.concatenate((train_1, train_2, train_3))
        elif mode == "val":
            idxes = np.loadtxt(f"data/idx_set_{(seed + 3) % 5}.csv", delimiter=",").astype("int")
        elif mode == "test":
            idxes = np.loadtxt(f"data/idx_set_{(seed + 4) % 5}.csv", delimiter=",").astype("int")
        self.data.data = self.data.data[idxes]
        self.plabels = self.plabels[idxes]
        self.len = self.plabels.shape[0]

        # Create dataloader
        self.data.targets = self.plabels
        self.dl = DataLoader(self.data, batch_size=batch_size, shuffle=(mode == "train"), num_workers=2)

        # Create tensorized versions
        self.t_data = torch.from_numpy(self.data.data).to(device)
        self.plabels = self.plabels.to(device)

        # Prepare negative sampling
        self.p_different_class = None

    def get_dataloader(self):
        # If we just want to loop over our data, e.g., for validation and test
        return self.dl

    def sample_x(self, n=64):
        # Return some random images, without labels
        ids = torch.multinomial(torch.ones(self.len, device=self.device), n, replacement=False)
        x = torch.stack([self.data.__getitem__(i)[0] for i in ids], dim=0).to(self.device)

        return x

    def sample(self, n=64, same_ref=False, n_repeat=1, n_neg=1):
        # For generating contrastive samples, similar to models/generator.py
        ref_ids = torch.multinomial(torch.ones(self.len, device=self.device), n, replacement=False)
        if same_ref:
            ref_ids[:,:] = ref_ids[0,:]
        if n_repeat > 1:
            ref_ids.repeat(n_repeat, 1)

        # generate pos and neg samples
        pos_ids = self._sample_pos_by_candidates(ref_ids)
        if n_neg > 0:
            neg_ids = self._sample_neg(ref_ids, n_neg)

        # cast ids to images
        x_ref = torch.stack([self.data.__getitem__(i)[0] for i in ref_ids], dim=0).to(self.device)
        x_pos = torch.stack([self.data.__getitem__(i)[0] for i in pos_ids], dim=0).to(self.device).unsqueeze(1)
        if n_neg > 0:
            x_neg = torch.stack([self.data.__getitem__(i)[0] for i in torch.flatten(neg_ids)], dim=0).to(self.device)
            x_neg = torch.reshape(x_neg, [*neg_ids.shape, *x_neg.shape[1:]])
        else:
            x_neg = None

        return x_ref, x_pos, x_neg

    def _sample_neg(self, ref_ids, n_neg=1):
        if self.p_different_class is None:
            # First time, calculate it:
            self.p_different_class = 1 - torch.matmul(self.plabels, self.plabels.t())

        batchsize = ref_ids.shape[0]

        # Generate candidates until each z_ref has a sample
        partner_ids = torch.zeros((batchsize, n_neg), device=self.device)
        needs_partner = torch.ones((batchsize, n_neg), dtype=torch.uint8, device=self.device)
        while torch.any(needs_partner):
            # Limit ourselves to those samples that need partners (for efficiency)
            requires_partner = torch.any(needs_partner, dim=1)

            # Sample whether other samples are neg to the ref
            is_ref_and_cand_wanted = torch.bernoulli(self.p_different_class[ref_ids[requires_partner]])
            is_ref_and_cand_wanted = is_ref_and_cand_wanted.type(torch.uint8)
            # Choose samples
            # in is_ref_and_cand_wanted we might have rows with full 0. This crashes torch.multinomial.
            # In case we have no 1, give everything a one and then filter out everything again afterwards
            p_select_bigger0 = is_ref_and_cand_wanted.float() + (torch.sum(is_ref_and_cand_wanted, dim=1) == 0).unsqueeze(1)
            chosen_idxes = torch.multinomial(p_select_bigger0, n_neg, replacement=False)

            # Choose the actual matches for each ref sample:
            for sub_idx, overall_idx in enumerate(requires_partner.nonzero()[:, 0]):
                # sub_idx is the index with respect to those that require a partner (the first that requires a partner, the second, ...)
                # overall_idx is the general idx of those samples (e.g., 8, 17, 52, ...)
                # The chosen_idx will probably contain samples with probability 0, because we forced it to sample n things,
                # even if there were less than n possible 1s in the array.
                n_matches = torch.sum(is_ref_and_cand_wanted[sub_idx])
                n_needed = torch.sum(needs_partner[overall_idx, :])
                n_new_samples = torch.min(n_matches, n_needed).type(torch.int)
                if n_new_samples > 0:
                    # One trick we can use is that the prob-0 choices are always at the end
                    chosen_idx = chosen_idxes[sub_idx, :n_new_samples]
                    partner_ids[overall_idx, n_neg - n_needed:(n_neg - n_needed + n_new_samples)] = chosen_idx
                    needs_partner[overall_idx, n_neg - n_needed:(n_neg - n_needed + n_new_samples)] = False

        # The dataloader expects int on cpu
        partner_ids = partner_ids.cpu().type(torch.uint8)
        return partner_ids

    def _sample_pos_by_candidates(self, ref_ids):
        batchsize = ref_ids.shape[0]
        id_partner = torch.zeros(batchsize, device=self.device).long()
        needs_partner = torch.ones(batchsize, dtype=torch.uint8, device=self.device)
        while torch.any(needs_partner):
            # Draw a class that we assume the reference belongs to
            ref_class = torch.multinomial(self.plabels[ref_ids], num_samples=1).squeeze(1)

            # See if we can find positive matches in that class
            is_ref_and_cand_pos = torch.bernoulli(self.plabels[:, ref_class].t())
            p_select_bigger0 = is_ref_and_cand_pos + (torch.sum(is_ref_and_cand_pos, dim=1) == 0).unsqueeze(1)
            chosen_idxes = torch.multinomial(p_select_bigger0, num_samples=1, replacement=False)

            n_matches = torch.sum(is_ref_and_cand_pos, dim=1)
            id_partner[torch.logical_and(needs_partner, n_matches > 0)] = chosen_idxes[torch.logical_and(needs_partner, n_matches > 0), 0]
            needs_partner[torch.logical_and(needs_partner, n_matches > 0)] = False

        return id_partner

    def get_data(self):
        return self.data


class ContrastiveCifarHard(ContrastiveCifar):
    def __init__(self, mode="train", seed=1, batch_size=64, device=torch.device("cuda:0")):
        super().__init__(mode=mode, seed=seed, batch_size=batch_size, device=device)

        # Make softlabels hard
        for i in torch.arange(self.plabels.shape[0]):
            hard_labels = torch.zeros(self.plabels.shape[1], device=self.device)
            hard_labels[torch.argmax(self.plabels[i,:])] = 1.
            self.plabels[i,:] = hard_labels

        # Create dataloader
        self.data.targets = self.plabels.cpu().numpy()
        self.dl = DataLoader(self.data, batch_size=batch_size, shuffle=(mode == "train"), num_workers=2)


class ContrastiveCifarHardTrain(ContrastiveCifar):
    def __init__(self, mode="train", batch_size=64, device=torch.device("cuda:0"), random_augs=False):
        super().__init__(mode=mode, batch_size=batch_size, device=device)

        # Load data
        if random_augs:
            self.transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
        self.data = datasets.CIFAR10(root='data/data_CIFAR10_test', train=True, download=True, transform=self.transform)
        self.plabels = torch.zeros((len(self.data.targets), 10))
        self.plabels.scatter_(dim=1, index=torch.Tensor(self.data.targets).type(torch.long).unsqueeze(1), value=1.)
        self.len = self.plabels.shape[0]

        # Create dataloader
        self.dl = DataLoader(self.data, batch_size=batch_size, shuffle=(mode == "train"), num_workers=2)

        # Create tensorized versions
        self.t_data = torch.from_numpy(self.data.data).to(device)
        self.plabels = self.plabels.to(device)


def make_lossy_dataloader(dataset, batchsize=64, shuffle=False):
    lossy_dataset = LossyCifar(dataset)
    return DataLoader(lossy_dataset, batch_size=batchsize, shuffle=shuffle, num_workers=2)

########################################################################################################################
# The following code is modified from https://github.com/tinkoff-ai/probabilistic-embeddings, Apache License 2.0       #
########################################################################################################################


class DatasetWrapper(torch.utils.data.Dataset):
    """Base class for dataset extension."""

    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        """Get base dataset."""
        return self._dataset

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return self.dataset.classification

    @property
    def openset(self):
        return self.dataset.openset

    @property
    def labels(self):
        """Get dataset labels array.
        Labels are integers in the range [0, N-1].
        """
        return self.dataset.labels

    @property
    def has_quality(self):
        """Whether dataset assigns quality score to each sample or not."""
        return self.dataset.has_quality

    def __len__(self):
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Get element of the dataset.
        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).
        Datasets with quality assigned to each sample return tuples like
        (image, label, quality) or ((image1, image2), label, (quality1, quality2)).
        """
        return self.dataset[index]


class LossyCifar(DatasetWrapper):
    """Add lossy transformations to input data."""
    def __init__(self, dataset):
        super().__init__(dataset)

        crop_min = 0.25
        crop_max = 1.0
        if crop_min > crop_max:
            raise AssertionError("Crop min size is greater than max.")
        # See if we already stored random crops for this configuration (to make them the same across runs)
        filepath = f'./data/randomcrops_{len(dataset)}_{crop_min}_{crop_max}.csv'
        if os.path.exists(filepath):
            self._center_crop = np.loadtxt(filepath, delimiter=",")
        else:
            self._center_crop = np.random.random(len(dataset)) * (crop_max - crop_min) + crop_min
            np.savetxt(filepath, self._center_crop, delimiter=",")

    @property
    def has_quality(self):
        """Whether dataset assigns quality score to each sample or not."""
        return True

    def __getitem__(self, index):
        """Get element of the dataset.
        Classification dataset returns tuple (image, soft_labels, quality).
        """
        image, label = self.dataset[index]

        if isinstance(image, Image.Image):
            image = np.asarray(image)

        center_crop = self._center_crop[index]
        if abs(center_crop - 1) > 1e-6:
            if isinstance(image, np.ndarray):
                # Image in HWC format.
                size = int(round(min(image.shape[0], image.shape[1]) * center_crop))
                y_offset = (image.shape[0] - size) // 2
                x_offset = (image.shape[1] - size) // 2
                image = image[y_offset:y_offset + size, x_offset:x_offset + size]
            elif isinstance(image, torch.Tensor):
                # Image in CHW format.
                size = int(round(min(image.shape[1], image.shape[2]) * center_crop))
                old_size = [image.shape[1], image.shape[2]]
                image = functional_transforms.center_crop(image, size)
                image = functional_transforms.resize(image, old_size)
            else:
                raise ValueError("Expected Numpy or torch Tensor.")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        quality = center_crop
        return image, label, quality
