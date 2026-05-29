import math
import torch
import torch.nn.functional as F

from offline.bcp import greedy_allocation
from cost import get_normalized_costs
from offline.lazy_threshold_bcp import th_greedy_allocation


@torch.no_grad()
def sampling_process(X_all, init_indices, costs, sample_size, sample_budget, delta, task_type, r=3.5e-7):

    _, best_gain, _ = th_greedy_allocation(
        X_all=X_all[:sample_size],
        init_indices=[idx for idx in init_indices if idx < sample_size],
        costs=costs[:sample_size],
        budget=sample_budget,
        epsilon=0.05,
        task_type=task_type,
        r=r
    )

    print(f'sample_budget: {sample_budget}')
    print(f'best_utility: {best_gain}')

    density_threshold = (best_gain / sample_budget) / delta

    print(f'density_threshold: {density_threshold}')

    return density_threshold


@torch.no_grad()
def multi_stage_mechanism(
    X_all,
    init_indices,
    costs,
    budget,
    density_threshold_init,
    task_type,
    r=3.5e-7
):

    N = X_all.size(0)

    density_threshold = density_threshold_init

    stage_budget = budget / int(2 ** math.log2(N))
    stage_size = int(N / int(2 ** math.log2(N)))

    total_price = 0

    payments = []
    gains= []

    W = []
    W_set = set()

    init_set = set(init_indices)

    # ========================================================
    # Image Classification
    # ========================================================

    if task_type == 'image_classification':

        X_all_norm = F.normalize(X_all, dim=1)

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

        current_utility = current_max_sim.sum().item()
        current_gain = 0

        for i in range(N):

            if i in init_set:
                continue

            x_i = X_all_norm[i:i+1]

            candidate_sim = (X_all_norm @ x_i.T).squeeze(1)

            updated_max = torch.maximum(
                current_max_sim,
                candidate_sim
            )

            gain = (
                updated_max - current_max_sim
            ).sum().item()

            if density_threshold == 0:
                density_threshold = density_threshold_init

            offer_price = gain / density_threshold

            if (
                costs[i] <= offer_price
                and offer_price <= (stage_budget - total_price)
            ):

                total_price += offer_price

                W.append(i)
                W_set.add(i)

                current_max_sim = updated_max

                current_utility += gain
                current_gain += gain

                payments.append(offer_price)
                gains.append(current_gain)

            if i == stage_size - 1 and i < N - 1:

                print(W)

                print(f'remaining-budget: {stage_budget - total_price}')
                print(f'stage-size: {stage_size}')

                density_threshold = sampling_process(
                    X_all,
                    init_indices,
                    costs,
                    stage_size,
                    stage_budget,
                    10,
                    task_type,
                    r
                )

                stage_size = min(N, stage_size * 2)
                stage_budget = min(budget, stage_budget * 2)

        return W, payments, gains

    # ========================================================
    # Crowdsensing
    # ========================================================

    elif task_type == 'crowdsensing':

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

                covered_mask |= (dist2 <= r * r)

        current_utility = covered_mask.sum().item()

        current_gain = 0

        for i in range(N):

            if i in init_set:
                continue

            x_i = X_all[i:i+1]

            diff = X_all - x_i

            dist2 = (diff * diff).sum(dim=1)

            covered = (dist2 <= r * r)

            new_cover = covered & (~covered_mask)

            gain = new_cover.sum().item()

            if density_threshold == 0:
                density_threshold = density_threshold_init

            offer_price = gain / density_threshold

            if (
                costs[i] <= offer_price
                and offer_price <= (stage_budget - total_price)
            ):

                total_price += offer_price

                W.append(i)
                W_set.add(i)

                covered_mask |= covered

                current_utility += gain
                current_gain += gain

                payments.append(offer_price)
                gains.append(current_gain)

            if i == stage_size - 1 and i < N - 1:

                print(W)

                print(f'remaining-budget: {stage_budget - total_price}')
                print(f'stage-size: {stage_size}')

                density_threshold = sampling_process(
                    X_all,
                    init_indices,
                    costs,
                    stage_size,
                    stage_budget,
                    10,
                    task_type,
                    r
                )

                stage_size = min(N, stage_size * 2)
                stage_budget = min(budget, stage_budget * 2)

        return W, payments, gains

    else:

        raise ValueError(f'Unknown task_type: {task_type}')