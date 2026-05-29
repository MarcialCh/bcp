import math

import torch

from cost import get_normalized_costs
from score import coverage_score, marginal_gain

import heapq






# ============================================================
# greedy Allocation
# ============================================================

import math
import heapq
import torch
import torch.nn.functional as F


@torch.no_grad()
def th_greedy_allocation(
    X_all,
    init_indices,
    costs,
    budget,
    epsilon,
    task_type,
    r=3.5e-7
):
    """
    Lazy Greedy Allocation

    Return:
        W
        current_utility
        trace
    """

    total_budget = budget

    N = X_all.size(0)

    W = []
    W_set = set()

    init_set = set(init_indices)

    densities = []

    trace = []

    updates = [0] * N

    max_update = math.log(N / epsilon) / epsilon

    # =========================================================
    # Image Classification
    # =========================================================

    if task_type == 'image_classification':

        # -----------------------------------------------------
        # normalize once
        # -----------------------------------------------------

        X_all_norm = F.normalize(X_all, dim=1)

        # -----------------------------------------------------
        # initialize current max similarity
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # initialize heap
        # -----------------------------------------------------

        for i in range(N):

            if costs[i] > total_budget:
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

            heapq.heappush(
                densities,
                (-density_i, i)
            )

        # =====================================================
        # lazy greedy loop
        # =====================================================

        while densities:

            neg_density, best_i = heapq.heappop(densities)

            best_density = -neg_density

            best_gain = best_density * costs[best_i]

            # -------------------------------------------------
            # recompute exact gain
            # -------------------------------------------------

            x_i = X_all_norm[best_i:best_i+1]

            candidate_sim = (
                X_all_norm @ x_i.T
            ).squeeze(1)

            updated_max = torch.maximum(
                current_max_sim,
                candidate_sim
            )

            exact_gain = (
                updated_max - current_max_sim
            ).sum().item()

            # -------------------------------------------------
            # lazy update
            # -------------------------------------------------

            if exact_gain < (1 - epsilon) * best_gain:

                new_density = exact_gain / costs[best_i]

                updates[best_i] += 1

                if updates[best_i] <= max_update:

                    heapq.heappush(
                        densities,
                        (-new_density, best_i)
                    )

                continue

            # -------------------------------------------------
            # proportional share
            # -------------------------------------------------

            proportional_share = (
                (1 - epsilon)
                * total_budget
                / 2
                * exact_gain
                / (current_gain + exact_gain)
            )

            if costs[best_i] > proportional_share:
                break

            # -------------------------------------------------
            # accept winner
            # -------------------------------------------------

            W.append(best_i)

            W_set.add(best_i)

            print(f'best_gain: {best_gain}')
            print(W)

            current_max_sim = updated_max

            current_gain += exact_gain

            current_utility += exact_gain

            trace.append({
                'winner': best_i,
                'gain': exact_gain,
                'utility': current_utility,
            })

        return W, current_gain, trace

    # =========================================================
    # Crowdsensing
    # =========================================================

    elif task_type == 'crowdsensing':

        # -----------------------------------------------------
        # initialize covered mask
        # -----------------------------------------------------

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

        current_utility = covered_mask.sum().item()

        current_gain = 0

        # -----------------------------------------------------
        # initialize heap
        # -----------------------------------------------------

        for i in range(N):

            if costs[i] > total_budget:
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

            heapq.heappush(
                densities,
                (-density_i, i)
            )

        # =====================================================
        # lazy greedy loop
        # =====================================================

        while densities:

            neg_density, best_i = heapq.heappop(densities)

            best_density = -neg_density

            best_gain = best_density * costs[best_i]

            # -------------------------------------------------
            # recompute exact gain
            # -------------------------------------------------

            x_i = X_all[best_i:best_i+1]

            diff = X_all - x_i

            dist2 = (diff * diff).sum(dim=1)

            covered = (dist2 <= r * r)

            new_cover = covered & (~covered_mask)

            exact_gain = new_cover.sum().item()

            # -------------------------------------------------
            # lazy update
            # -------------------------------------------------

            if exact_gain < (1 - epsilon) * best_gain:

                new_density = exact_gain / costs[best_i]

                updates[best_i] += 1

                if updates[best_i] <= max_update:

                    heapq.heappush(
                        densities,
                        (-new_density, best_i)
                    )

                continue

            # -------------------------------------------------
            # proportional share
            # -------------------------------------------------

            proportional_share = (
                (1 - epsilon)
                * total_budget
                / 2
                * exact_gain
                / (current_gain + exact_gain)
            )

            if costs[best_i] > proportional_share:
                break

            # -------------------------------------------------
            # accept winner
            # -------------------------------------------------

            W.append(best_i)

            W_set.add(best_i)

            print(f'best_gain: {best_gain}')
            print(W)

            covered_mask |= covered

            current_gain += exact_gain

            current_utility += exact_gain

            trace.append({
                'winner': best_i,
                'gain': exact_gain,
                'utility': current_utility,
            })

        return W, current_gain, trace

    else:
        raise ValueError(
            f'Unknown task_type: {task_type}'
        )


# ============================================================
# threshold Payment
# ============================================================

def compute_payments(X_all, costs, W, budget, epsilon, task_type):

    payments = {}

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

        W_minus_k, _, trace_minus_k = th_greedy_allocation(X_minus_k, costs_minus_k, budget, task_type)

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

            gain_k = marginal_gain(X_all, S, k, task_type)

            if gain_k <= 0:
                continue

            # ------------------------------------------------
            # utility with k
            # ------------------------------------------------

            indices_Sk = torch.tensor(S + [k], device=X_all.device)

            utility_with_k = coverage_score(X_all, X_all[indices_Sk], task_type)

            # ------------------------------------------------
            # last virtual position
            # ------------------------------------------------

            if j == len(trace_minus_k):

                beta_kj = (1 - epsilon) * (budget / 2 * gain_k / utility_with_k)

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

            beta_kj = (1 - epsilon) * (budget / 2 * gain_k / utility_with_k)

            candidate_payment = min(b_kj,
                beta_kj
            )

            payment_k = max(payment_k, candidate_payment)

        payments[k] = payment_k

    return payments


# ============================================================
# main mechanism
# ============================================================

def budget_feasible_mechanism(X_all, budget, task_type):

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

        if costs[i] > budget:
            continue

        utility_i = coverage_score(X_all, X_all[i].unsqueeze(0), task_type)

        if utility_i > singleton_utility:
            singleton_utility = utility_i
            singleton_winner = i

    # --------------------------------------------------------
    # greedy solution
    # --------------------------------------------------------

    W_greedy, trace = th_greedy_allocation(X_all, costs, budget, task_type)

    if len(W_greedy) == 0:

        return (
            [singleton_winner],
            {singleton_winner: budget},
        )

    indices_greedy = torch.tensor(W_greedy, device=X_all.device)

    greedy_utility = coverage_score(
        X_all,
        X_all[indices_greedy],
        task_type
    )

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
        costs,
        W_greedy,
        budget,
        task_type
    )

    return W_greedy, payments


        

