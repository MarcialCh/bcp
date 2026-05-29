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
from online.ms import multi_stage_mechanism
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

    budgets = list(range(100, 701, 100))

    all_results = []
    all_scores = []

    X_all = flatten_images(trainset)
    # X_all = resample(X_all, n_samples=5000, replace=False, random_state=42)


    num_runs = 5

    label_num = 3

    # target_labels = [0, 1] # [airplane, automobile]
    # target_labels = [0, 1, 2] # [airplane, automobile, bird]

    # task = 'offline_lz'

    task = 'offline_bcp'

    # task = 'offline_greedy'

    epsilon = 0.1



    orig_train_data = trainset.data.copy()
    orig_train_targets = np.array(trainset.targets)

    orig_test_data = testset.data.copy()
    orig_test_targets = np.array(testset.targets)


    target_labels = list(range(label_num))

    train_mask = np.isin(orig_train_targets, target_labels)

    trainset.data = orig_train_data[train_mask]
    trainset.targets = orig_train_targets[train_mask].tolist()

    test_mask = np.isin(orig_test_targets, target_labels)

    testset.data = orig_test_data[test_mask]
    testset.targets = orig_test_targets[test_mask].tolist()

    X_all = flatten_images(trainset)

    init_indices =  np.random.choice(X_all.shape[0], size=30, replace=False)



    for budget in budgets:

        print(f"\n==============================")
        print(f"Budget: {budget}")
        print(f"==============================")

        acc_list = []
        score_list = []

        for run in range(num_runs):

            seed = run
            print(f"\n--- Run {run+1}/{num_runs}, seed={seed} ---")

            f = open('/data/usrs/yyr/cc/pricing/image_classification/cifar10' + "/" + str(len(target_labels)) + "_" + str(task) + "_" + str(epsilon) + "_" + str(budget), 'r')
            lines = f.readlines()
            winners = [int(v) for v in lines[0].split()]
            trainloader, testloader = get_dataloader(trainset, testset, batch_size=128, indices = winners + list(init_indices))

            model = ResNet56(num_classes=label_num).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=5e-4
            )

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[60, 80], gamma=0.1
            )

            # ===== training =====
            for epoch in range(1, 101):
                loss, train_acc = train_one_epoch(
                    model, trainloader, optimizer, criterion, device
                )
                scheduler.step()

            # ===== evaluation =====
            test_acc = evaluate(model, testloader, device)

            # subset_indices = trainloader.dataset.indices
            # X_subset = X_all[subset_indices]

            # score = coverage_score(X_all, X_subset, 'image_classification')
            # score = summarization_score(X_all, X_subset)

            print(f"Run {run}: acc={test_acc:.2f}")

            acc_list.append(test_acc)

        # ===== mean over runs =====
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)


        all_results.append(mean_acc)

        print(f"\n>>> Budget {budget} FINAL")
        print(f"Accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
    # =========================
    # final report
    # =========================
    print("\n==== FINAL SUMMARY ====")

    for budget, acc in zip(budgets, all_results):
        print(f"{budget}: acc={acc:.2f}")


if __name__ == "__main__":
    main()