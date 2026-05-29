import numpy as np
import math
import random
import torch
import torch.nn.functional as F

from cost import get_normalized_costs


@torch.no_grad()
def sampling_process(X_all, init_indices, costs, budget, task_type, r=3.5e-7):

    N = X_all.size(0)

    tau = np.random.binomial(N, 0.5)

    max_gain = 0

    init_set = set(init_indices)

    # ========================================================
    # Image Classification
    # ========================================================

    if task_type == 'image_classification':

        X_all_norm = F.normalize(X_all, dim=1)

        # ----------------------------------------------------
        # initialize current max similarity
        # ----------------------------------------------------

        if len(init_indices) > 0:

            init_tensor = torch.tensor(
                init_indices,
                device=X_all.device
            )

            init_subset = X_all_norm[init_tensor]

            sim = X_all_norm @ init_subset.T

            current_max_sim = sim.max(dim=1).values

        else:

            current_max_sim = torch.zeros(
                N,
                device=X_all.device
            )

        # ----------------------------------------------------
        # compute max marginal gain
        # ----------------------------------------------------

        for i in range(tau):

            if costs[i] > budget or i in init_set:
                continue

            x_i = X_all_norm[i:i+1]

            candidate_sim = (
                X_all_norm @ x_i.T
            ).squeeze(1)

            updated_max = torch.maximum(
                current_max_sim,
                candidate_sim
            )

            gain = (
                updated_max - current_max_sim
            ).sum().item()

            max_gain = max(max_gain, gain)

    # ========================================================
    # Crowdsensing
    # ========================================================

    elif task_type == 'crowdsensing':

        # ----------------------------------------------------
        # initialize covered mask
        # ----------------------------------------------------

        covered_mask = torch.zeros(
            N,
            dtype=torch.bool,
            device=X_all.device
        )

        if len(init_indices) > 0:

            init_tensor = torch.tensor(
                init_indices,
                device=X_all.device
            )

            init_subset = X_all[init_tensor]

            for x in init_subset:

                diff = X_all - x

                dist2 = (diff * diff).sum(dim=1)

                covered_mask |= (
                    dist2 <= r * r
                )

        # ----------------------------------------------------
        # compute max marginal gain
        # ----------------------------------------------------

        for i in range(tau):

            if costs[i] > budget or i in init_set:
                continue

            x_i = X_all[i:i+1]

            diff = X_all - x_i

            dist2 = (diff * diff).sum(dim=1)

            covered = (dist2 <= r * r)

            new_cover = covered & (~covered_mask)

            gain = new_cover.sum().item()

            max_gain = max(max_gain, gain)

    else:

        raise ValueError(
            f'Unknown task_type: {task_type}'
        )

    l = random.randint(0, math.ceil(math.log2(N)))

    density_threshold = (budget/ ((2 ** l) * max_gain))

    return tau, density_threshold


@torch.no_grad()
def online_post_mechanism(X_all, init_indices, costs, budget, task_type, r=3.5e-7):

    N = X_all.size(0)

    remaining_budget = budget

    W = []

    payments = []

    gains = []

    init_set = set(init_indices)

    # ========================================================
    # Sampling
    # ========================================================

    tau, density_threshold = sampling_process(X_all, init_indices, costs, budget, task_type, r)

    # ========================================================
    # Image Classification
    # ========================================================

    if task_type == 'image_classification':

        X_all_norm = F.normalize(X_all, dim=1)

        # ----------------------------------------------------
        # initialize current max similarity
        # ----------------------------------------------------

        if len(init_indices) > 0:

            init_tensor = torch.tensor(
                init_indices,
                device=X_all.device
            )

            init_subset = X_all_norm[init_tensor]

            sim = X_all_norm @ init_subset.T

            current_max_sim = sim.max(dim=1).values

        else:

            current_max_sim = torch.zeros(
                N,
                device=X_all.device
            )

        # ----------------------------------------------------
        # online selection
        # ----------------------------------------------------

        for i in range(tau, N):

            if i in init_set:
                continue

            x_i = X_all_norm[i:i+1]

            candidate_sim = (
                X_all_norm @ x_i.T
            ).squeeze(1)

            updated_max = torch.maximum(
                current_max_sim,
                candidate_sim
            )

            gain = (
                updated_max - current_max_sim
            ).sum().item()

            offer_price = gain / density_threshold

            if costs[i] <= offer_price and offer_price <= remaining_budget:

                W.append(i)

                payments.append(
                    offer_price
                )

                gains.append(gain)

                remaining_budget -= offer_price

                current_max_sim = updated_max

        return W, payments, gains

    # ========================================================
    # Crowdsensing
    # ========================================================

    elif task_type == 'crowdsensing':

        # ----------------------------------------------------
        # initialize covered mask
        # ----------------------------------------------------

        covered_mask = torch.zeros(
            N,
            dtype=torch.bool,
            device=X_all.device
        )

        # init coverage
        if len(init_indices) > 0:

            init_tensor = torch.tensor(
                init_indices,
                device=X_all.device
            )

            init_subset = X_all[init_tensor]

            for x in init_subset:

                diff = X_all - x

                dist2 = (diff * diff).sum(dim=1)

                covered_mask |= (
                    dist2 <= r * r
                )

        # ----------------------------------------------------
        # online selection
        # ----------------------------------------------------

        for i in range(tau, N):

            if i in init_set:
                continue

            x_i = X_all[i:i+1]

            diff = X_all - x_i

            dist2 = (diff * diff).sum(dim=1)

            covered = (dist2 <= r * r)

            new_cover = covered & (~covered_mask)

            gain = new_cover.sum().item()

            offer_price = gain / density_threshold

            if costs[i] <= offer_price and offer_price <= remaining_budget:

                W.append(i)

                payments.append(
                    offer_price
                )

                gains.append(gain)

                remaining_budget -= offer_price

                covered_mask |= covered

        return W, payments, gains

    else:

        raise ValueError(
            f'Unknown task_type: {task_type}'
        )

