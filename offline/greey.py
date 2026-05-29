

from cost import get_normalized_costs
from score import coverage_score, marginal_gain


import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_mechanism(
    X_all,
    init_indices,
    costs,
    budget,
    task_type,
    r=3.5e-4
):

    N = X_all.size(0)

    W = []
    W_set = set()

    init_set = set(init_indices)

    payments = []

    gains = []

    trace = []

    remaining_budget = budget/2

    # ========================================================
    # Image Classification
    # ========================================================

    if task_type == 'image_classification':

        # ----------------------------------------------------
        # normalize once
        # ----------------------------------------------------

        X_all_norm = F.normalize(
            X_all,
            dim=1
        )

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

        current_utility = (
            current_max_sim.sum().item()
        )

        current_gain = 0

        # ----------------------------------------------------
        # initialize density list
        # ----------------------------------------------------

        densities = []

        for i in range(N):

            if costs[i] > budget:
                continue

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

            gain_i = (
                updated_max - current_max_sim
            ).sum().item()

            density_i = gain_i / costs[i]

            densities.append((i, density_i))

        # ----------------------------------------------------
        # sort by density
        # ----------------------------------------------------

        densities.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ----------------------------------------------------
        # greedy selection
        # ----------------------------------------------------

        for index, _ in densities:

            if costs[index] > remaining_budget:
                continue

            x_i = X_all_norm[index:index+1]

            candidate_sim = (
                X_all_norm @ x_i.T
            ).squeeze(1)

            updated_max = torch.maximum(
                current_max_sim,
                candidate_sim
            )

            greedy_gain = (
                updated_max - current_max_sim
            ).sum().item()

            if greedy_gain <= 0:
                continue

            remaining_budget -= costs[index]

            W.append(index)

            W_set.add(index)

            payments.append(costs[index])

            gains.append(greedy_gain)

            current_max_sim = updated_max

            current_utility += greedy_gain

            current_gain += greedy_gain

            trace.append({
                'winner': index,
                'gain': greedy_gain,
                'utility': current_utility,
            })

        return W, current_gain, trace

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

                covered_mask |= (dist2 <= r * r)

        current_utility = (
            covered_mask.sum().item()
        )

        current_gain = 0

        # ----------------------------------------------------
        # initialize density list
        # ----------------------------------------------------

        densities = []

        for i in range(N):

            if costs[i] > budget:
                continue

            if i in init_set:
                continue

            x_i = X_all[i:i+1]

            diff = X_all - x_i

            dist2 = (diff * diff).sum(dim=1)

            covered = (dist2 <= r * r)

            new_cover = covered & (~covered_mask)

            gain_i = new_cover.sum().item()

            density_i = gain_i / costs[i]

            densities.append((i, density_i))

        # ----------------------------------------------------
        # sort by density
        # ----------------------------------------------------

        densities.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ----------------------------------------------------
        # greedy selection
        # ----------------------------------------------------

        for index, _ in densities:

            if costs[index] > remaining_budget:
                continue

            x_i = X_all[index:index+1]

            diff = X_all - x_i

            dist2 = (diff * diff).sum(dim=1)

            covered = (dist2 <= r * r)

            new_cover = covered & (~covered_mask)

            greedy_gain = new_cover.sum().item()

            if greedy_gain <= 0:
                continue

            remaining_budget -= costs[index]

            W.append(index)

            W_set.add(index)

            payments.append(costs[index])

            gains.append(greedy_gain)

            covered_mask |= covered

            current_utility += greedy_gain

            current_gain += greedy_gain

            trace.append({
                'winner': index,
                'gain': greedy_gain,
                'utility': current_utility,
            })

        return W, current_gain, trace

    else:

        raise ValueError(
            f'Unknown task_type: {task_type}'
        )


