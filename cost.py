# cost.py

import torch
import numpy as np


def compute_labeling_costs(dataset):

    """
    cost ∝ std(pixel intensities)

    return:
        costs: numpy array [N]
    """

    data = dataset.data

    # ensure tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    data = data.float()

    # flatten each image
    data = data.view(data.size(0), -1)

    # std per image
    costs = torch.std(data, dim=1)

    return costs.numpy()


def normalize_costs(costs):
    """
    normalize bids so that mean = 1
    """

    mean_cost = np.mean(costs)

    normalized = costs / mean_cost

    return normalized


def get_normalized_costs(dataset, task_type):
    N = dataset.size(0)

    if task_type == 'image_classification':
        costs = compute_labeling_costs(dataset)
        normalized_costs = normalize_costs(costs)
        # normalized_costs = [1]*N


    elif task_type == 'crowdsensing':
        normalized_costs = [1]*N



    return normalized_costs