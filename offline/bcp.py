import torch

from cost import get_normalized_costs
from score import coverage_score, marginal_gain








# ============================================================
# greedy Allocation
# ============================================================

import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_allocation(
    X_all,
    init_indices,
    costs,
    budget,
    task_type,
    N=0,
    r=3.5e-7
):
    """
    Optimized Greedy Allocation

    Return:
        W
        current_utility
        trace
    """

    total_budget = budget

    if N == 0:
        N = X_all.size(0)

    W = []
    trace = []

    # ============================================================
    # Image Classification
    # ============================================================

    if task_type == 'image_classification':

        # --------------------------------------------------------
        # normalize once
        # --------------------------------------------------------

        X_all_norm = F.normalize(X_all, dim=1)

        # --------------------------------------------------------
        # initialize current_max_sim
        # --------------------------------------------------------

        if len(init_indices) > 0:

            init_tensor = torch.tensor(
                init_indices,
                device=X_all.device
            )

            init_subset = X_all_norm[init_tensor]

            # [N, |S|]
            sim = X_all_norm @ init_subset.T

            # [N]
            current_max_sim = sim.max(dim=1).values

            current_utility = current_max_sim.sum().item()

        else:

            current_max_sim = torch.zeros(
                X_all.size(0),
                device=X_all.device
            )

            current_utility = 0.0

        current_gain = 0.0

        # ========================================================
        # greedy loop
        # ========================================================

        while True:

            best_i = None
            best_gain = 0.0
            best_density = 0.0
            best_new_max = None

            for i in range(N):

                if i in W or i in init_indices:
                    continue

                # ------------------------------------------------
                # incremental marginal gain
                # ------------------------------------------------

                x_i = X_all_norm[i:i+1]

                # [N]
                candidate_sim = (
                    X_all_norm @ x_i.T
                ).squeeze(1)

                # [N]
                updated_max = torch.maximum(
                    current_max_sim,
                    candidate_sim
                )

                gain = (
                    updated_max - current_max_sim
                ).sum().item()

                if gain <= 0:
                    continue

                density = gain / costs[i]

                if density > best_density:

                    best_density = density
                    best_gain = gain
                    best_i = i
                    best_new_max = updated_max

            if best_i is None:
                break

            proportional_share = (
                total_budget / 2
                * best_gain
                / (current_gain + best_gain)
            )

            # proportional-share feasibility
            if costs[best_i] > proportional_share:
                break

            # ----------------------------------------------------
            # update
            # ----------------------------------------------------

            W.append(best_i)

            current_gain += best_gain
            current_utility += best_gain

            current_max_sim = best_new_max

            print(f'best_gain: {best_gain}')
            print(W)

            trace.append({
                'winner': best_i,
                'gain': best_gain,
                'utility': current_utility,
            })

        return W, current_gain, trace

    # ============================================================
    # Crowdsensing
    # ============================================================

    elif task_type == 'crowdsensing':

    # ========================================================
    # initialize
    # ========================================================

        W = []
        W_set = set()

        init_set = set(init_indices)

        # --------------------------------------------------------
        # covered mask
        # --------------------------------------------------------

        covered_mask = torch.zeros(
            X_all.size(0),
            dtype=torch.bool,
            device=X_all.device
        )

        # --------------------------------------------------------
        # initialize coverage from init set
        # --------------------------------------------------------

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

        # ========================================================
        # greedy loop
        # ========================================================

        while True:

            best_i = None
            best_gain = 0
            best_density = 0
            best_cover = None

            for i in range(N):

                if i in W_set or i in init_set:
                    continue

                # ------------------------------------------------
                # incremental marginal gain
                # ------------------------------------------------

                gain, covered = marginal_gain(
                    X_all=X_all,
                    W_set=W_set,
                    i=i,
                    task_type='crowdsensing',
                    covered_mask=covered_mask,
                    r=r
                )

                if gain <= 0:
                    continue

                density = gain / costs[i]

                if density > best_density:

                    best_density = density
                    best_gain = gain
                    best_i = i
                    best_cover = covered

            # ----------------------------------------------------
            # no feasible candidate
            # ----------------------------------------------------

            if best_i is None:
                break

            # ----------------------------------------------------
            # proportional share constraint
            # ----------------------------------------------------

            proportional_share = (
                total_budget / 2
                * best_gain
                / (current_gain + best_gain)
            )

            if costs[best_i] > proportional_share:
                break

            # ----------------------------------------------------
            # update
            # ----------------------------------------------------

            W.append(best_i)

            W_set.add(best_i)

            covered_mask |= best_cover

            current_gain += best_gain

            current_utility += best_gain

            print(f'best_gain: {best_gain}')

            print(W)

            trace.append({
                'winner': best_i,
                'gain': best_gain,
                'utility': current_utility,
            })

        return W, current_gain, trace

    else:
        raise ValueError(f'Unknown task_type: {task_type}')


# ============================================================
# threshold Payment
# ============================================================

def compute_payments(X_all, init_indices, costs, W, budget, task_type):

    payments = {}

    init_utility = coverage_score(X_all, X_all[init_indices], task_type)

    for k in W:

        # ----------------------------------------------------
        # Remove k
        # ----------------------------------------------------

        mask = [i for i in range(X_all.size(0)) if i != k]

        X_minus_k = X_all[mask]

        costs_minus_k = costs[mask]

        # mapping
        idx_map = {
            new_idx: old_idx
            for new_idx, old_idx in enumerate(mask)
        }

        # ----------------------------------------------------
        # re-run greedy without k
        # ----------------------------------------------------

        W_minus_k, _, trace_minus_k = greedy_allocation(X_minus_k, init_indices, costs_minus_k, budget, task_type)

        payment_k = 0

        S = []

        # ----------------------------------------------------
        # iterate each position
        # ----------------------------------------------------

        for j in range(len(trace_minus_k) + 1):

            # ------------------------------------------------
            # construct S_{j-1}
            # ------------------------------------------------

            if j > 0:

                selected_local = trace_minus_k[j - 1]['winner']

                selected_global = idx_map[selected_local]

                S.append(selected_global)

            # ------------------------------------------------
            # marginal gain of k
            # ------------------------------------------------

            gain_k = marginal_gain(X_all, S + init_indices, k, task_type)

            if gain_k <= 0:
                continue

            # ------------------------------------------------
            # utility with k
            # ------------------------------------------------

            indices_Sk = torch.tensor(S + init_indices + [k], device=X_all.device)

            utility_with_k = coverage_score(X_all, X_all[indices_Sk], task_type) - init_utility

            # ------------------------------------------------
            # last virtual position
            # ------------------------------------------------

            if j == len(trace_minus_k):

                beta_kj = (budget / 2 * gain_k / utility_with_k)

                payment_k = max(payment_k, beta_kj)

                continue

            # ------------------------------------------------
            # current seller j
            # ------------------------------------------------

            seller_local = trace_minus_k[j]['winner']

            seller_global = idx_map[seller_local]

            gain_j = trace_minus_k[j]['gain']

            bid_j = costs[seller_global]

            # ------------------------------------------------
            # replacement bid
            # ------------------------------------------------

            b_kj = (bid_j * gain_k / gain_j)

            # ------------------------------------------------
            # proportional-share threshold
            # ------------------------------------------------

            beta_kj = (budget / 2 * gain_k / utility_with_k)

            candidate_payment = min(b_kj,
                beta_kj
            )

            payment_k = max(payment_k, candidate_payment)

        payments[k] = payment_k

    return payments


# ============================================================
# main mechanism
# ============================================================

def budget_feasible_mechanism(X_all, init_indices, budget, task_type):

    costs = get_normalized_costs(
        X_all,
        task_type
    )

    # --------------------------------------------------------
    # singleton solution
    # --------------------------------------------------------

    singleton_winner = None
    singleton_utility = 0

    for i in range(X_all.size(0)):

        if costs[i] > budget or i in init_indices:
            continue

        utility_i = coverage_score(X_all, X_all[i].unsqueeze(0), task_type)

        if utility_i > singleton_utility:
            singleton_utility = utility_i
            singleton_winner = i

    # --------------------------------------------------------
    # greedy solution
    # --------------------------------------------------------

    W_greedy, _, _ = greedy_allocation(X_all, init_indices, costs, budget, task_type)

    if len(W_greedy) == 0:

        return (
            [singleton_winner],
            {singleton_winner: budget},
        )


    init_utility = coverage_score(X_all, X_all[init_indices], task_type)

    greedy_utility = coverage_score(X_all, X_all[W_greedy + init_indices], task_type) - init_utility

    # --------------------------------------------------------
    # choose better solution
    # --------------------------------------------------------

    if singleton_utility >= greedy_utility:

        return (
            [singleton_winner],
            {singleton_winner: budget},
        )

    # --------------------------------------------------------
    # compute payments
    # --------------------------------------------------------

    payments = compute_payments(
        X_all,
        init_indices,
        costs,
        W_greedy,
        budget,
        task_type
    )

    return W_greedy, payments


        