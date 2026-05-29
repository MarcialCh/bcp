# train.py

import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.utils import resample

from cost import get_normalized_costs
from dataloader import load_cifar10, get_dataloader
from model import ResNet56
from offline.bcp import greedy_allocation
from offline.greey import greedy_mechanism
from offline.lazy_threshold_bcp import th_greedy_allocation
from online.bid import online_bid_mechanism
from online.ms import multi_stage_mechanism
from online.post import online_post_mechanism
from online.util import summarize_intervals
from score import coverage_score, flatten_images


# =========================
# training
# =========================
def train_one_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss, 100. * correct / total


# =========================
# evaluation
# =========================
def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


# =========================
# main
# =========================
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(0)
    np.random.seed(0)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    trainset, testset = load_cifar10()

    budgets = list(range(100, 1001, 100))

    all_results = []
    all_scores = []

    X_all = flatten_images(trainset)
    # X_all = resample(X_all, n_samples=5000, replace=False, random_state=42)


    num_runs = 5

    r = 0

    label_num = 3

    # target_labels = [0, 1] # [airplane, automobile]
    # target_labels = [0, 1, 2] # [airplane, automobile, bird]

    task = 'online_ms'

    # task = 'online_bid'

    # task = 'online_post'

    epsilon = 0.05



    orig_train_data = trainset.data.copy()
    orig_train_targets = np.array(trainset.targets)

    orig_test_data = testset.data.copy()
    orig_test_targets = np.array(testset.targets)

    for label_num in [3]:

        target_labels = list(range(label_num))

        train_mask = np.isin(orig_train_targets, target_labels)

        trainset.data = orig_train_data[train_mask]
        trainset.targets = orig_train_targets[train_mask].tolist()

        test_mask = np.isin(orig_test_targets, target_labels)

        testset.data = orig_test_data[test_mask]
        testset.targets = orig_test_targets[test_mask].tolist()

        X_all = flatten_images(trainset)

        print(f'size:{X_all.size(0)}')

        init_indices =  np.random.choice(X_all.shape[0], size=30, replace=False)

        for budget in budgets:

            start = time.perf_counter()

            costs = get_normalized_costs(X_all, 'image_classification')

            if task == 'online_ms':

                winners, payments, gains = multi_stage_mechanism(X_all=X_all, 
                                                             init_indices=list(init_indices), 
                                                             costs=costs, 
                                                             budget=budget, density_threshold_init=10, task_type='crowdsensing', r=r)

            if task == 'online_post':
                winners, payments, gains = online_post_mechanism(X_all=X_all, 
                                                                init_indices=list(init_indices), 
                                                                costs=costs, budget=budget,  task_type='crowdsensing', r=r)
                
            if task == 'online_bid':
                winners, payments, gains = online_bid_mechanism(X_all=X_all,
                                                                init_indices=list(init_indices),
                                                                costs=costs,
                                                                budget=budget,
                                                                delta=10,
                                                                task_type = 'crowdsensing', r=r)


            end = time.perf_counter()

            running_time = end - start

            print(f"time: {running_time:.6f}s")

            N = X_all.size(0)
            m = 4
            winner_counts, cumulative_payment, cumulative_gain = summarize_intervals(winners, payments, gains, N, m)


            write_file = open('/data/usrs/yyr/cc/pricing/image_classification/online/cifar10' + "/" + str(task) + "_" + str(budget), 'w')
            write_file.write(' '.join([str(v) for v in winner_counts]) + "\n")
            write_file.write(' '.join([str(p) for p in cumulative_payment]) + "\n")
            write_file.write(' '.join([str(g) for g in cumulative_gain]) + "\n")
            # write_file.write(' '.join([str(u) for u in utility]) + "\n")
            write_file.flush()
            write_file.close()  


if __name__ == "__main__":
    main()