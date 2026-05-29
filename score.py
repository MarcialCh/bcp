# similarity_score.py

import math

import torch
import torch.nn.functional as F


def compute_cosine_similarity_matrix(X, Y):
    """
    X: [N, d]
    Y: [M, d]
    return: [N, M]
    """
    X = F.normalize(X, dim=1)
    Y = F.normalize(Y, dim=1)

    return torch.mm(X, Y.t())


def flatten_images(dataset):
    """
    flatten a CIFAR image into a 3072-dimensional vector
    """
    data = dataset.data  # (N, 32, 32, 3)
    X = torch.tensor(data).float().reshape(len(data), -1)  # (N, 3072)
    return X


import torch
import torch.nn.functional as F


# ============================================================
# Coverage Functions
# ============================================================

@torch.no_grad()
def image_coverage_score(
    X_all,
    X_subset,
    batch_size=2048,
    normalized=False
):
    """
    f(S) = sum_{u in N} max_{v in S} sim(u,v)

    X_all:    [N, d]
    X_subset: [M, d]
    """

    if X_subset.size(0) == 0:
        return 0.0

  
    if not normalized:
        X_all = F.normalize(X_all, dim=1)
        X_subset = F.normalize(X_subset, dim=1)

    total_score = 0.0

  
    for i in range(0, X_all.size(0), batch_size):

        batch = X_all[i:i + batch_size]          # [B, d]

        # [B, M]
        sim = batch @ X_subset.T

        # [B]
        max_sim = sim.max(dim=1).values

        total_score += max_sim.sum().item()

    return total_score


@torch.no_grad()
def crowdsensing_coverage_score(
    X_all,
    X_subset,
    r=3.5e-7,
    batch_size=4096
):
    """
    coverage count within radius r

    X_all:    [N, 2]
    X_subset: [M, 2]
    """

    if X_subset.size(0) == 0:
        return 0

    r2 = r * r
    covered_count = 0

    
    for i in range(0, X_all.size(0), batch_size):

        batch = X_all[i:i + batch_size]          # [B, 2]

        # [M, B, 2]
        diff = X_subset[:, None, :] - batch[None, :, :]

        # [M, B]
        dist2 = (diff * diff).sum(dim=-1)

        # [B]
        covered = (dist2 <= r2).any(dim=0)

        covered_count += covered.sum().item()

    return covered_count


def coverage_score(
    X_all,
    X_subset,
    task_type,
    **kwargs
):

    if task_type == 'image_classification':

        return image_coverage_score(
            X_all,
            X_subset,
            **kwargs
        )

    elif task_type == 'crowdsensing':

        return crowdsensing_coverage_score(
            X_all,
            X_subset,
            **kwargs
        )

    else:
        raise ValueError(f'Unknown task_type: {task_type}')


# ============================================================
# Marginal Gain
# ============================================================

@torch.no_grad()
def marginal_gain(
    X_all,
    W_set,
    i,
    task_type,
    covered_mask=None,
    current_max_sim=None,
    r=3.5e-7
):


    if i in W_set:
        return 0.0

    x_i = X_all[i:i + 1]

    # ========================================================
    # Image Classification
    # ========================================================

    if task_type == 'image_classification':

        X_all_norm = F.normalize(X_all, dim=1)
        x_i_norm = F.normalize(x_i, dim=1)

        # empty
        if current_max_sim is None:

            new_max_sim = (X_all_norm @ x_i_norm.T).squeeze(1)

            return new_max_sim.sum().item()

        # increment
        candidate_sim = (X_all_norm @ x_i_norm.T).squeeze(1)

        updated_max = torch.maximum(
            current_max_sim,
            candidate_sim
        )

        gain = (updated_max - current_max_sim).sum()

        return gain.item()

    # ========================================================
    # Crowdsensing
    # ========================================================

    elif task_type == 'crowdsensing':

        x_i = X_all[i:i + 1]

        diff = X_all - x_i

        dist2 = (diff * diff).sum(dim=1)

        covered = (dist2 <= r * r)

        # current set is empty
        if covered_mask is None:

            gain = covered.sum().item()

            return gain, covered

        # incremental coverage
        new_cover = covered & (~covered_mask)

        gain = new_cover.sum().item()

        return gain, covered

    else:
        raise ValueError(f'Unknown task_type: {task_type}')