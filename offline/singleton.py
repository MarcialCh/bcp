from cost import get_normalized_costs
from score import coverage_score

def get_singleton_solution(X_all, indices, budget, task_type):

    N = X_all.size(0)

    max_score = float('-inf')
    winner = None

    costs = get_normalized_costs(X_all, task_type)

    for i in range(N):

        if costs[i] > budget or i in indices:
            continue

        x_singleton = X_all[i].unsqueeze(0)

        score = coverage_score(X_all, x_singleton, task_type)

        if score > max_score:
            winner = i
            max_score = score

    return [winner], [budget], max_score

        

