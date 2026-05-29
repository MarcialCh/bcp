# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.utils import resample

from dataloader import load_cifar10, get_dataloader, load_stl10
from model import ResNet56
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

    trainset, testset = load_stl10()

    budgets = list(range(1000, 5001, 500))

    all_results = []
    all_scores = []

    X_all = flatten_images(trainset)
    # X_all = resample(X_all, n_samples=5000, replace=False, random_state=42)


    num_runs = 5

    target_labels = [3, 5]


    # labels = np.array(trainset.targets)   # CIFAR-10

    labels = np.array(trainset.labels)

    mask = np.isin(labels, target_labels)

    trainset.data = trainset.data[mask]
    trainset.labels = labels[mask]

    X_all = flatten_images(trainset)

    for budget in budgets:
        init_indices =  np.random.choice(X_all.shape[0], size=100, replace=False)

        winners, payments, utility = multi_stage_mechanism(X_all, list(init_indices), budget, 0.1, 'image_classification')

        write_file = open('/data/usrs/yyr/cc/pricing/image_classification/stl10'+ "/" + str(budget), 'w')
        write_file.write(' '.join([str(v) for v in winners]) + "\n")
        write_file.write(' '.join([str(p) for p in payments]) + "\n")
        write_file.write(' '.join([str(u) for u in utility]) + "\n")
        write_file.flush()
        write_file.close()   

    for budget in budgets:

        print(f"\n==============================")
        print(f"Budget: {budget}")
        print(f"==============================")

        acc_list = []
        score_list = []

        for run in range(num_runs):

            seed = run
            print(f"\n--- Run {run+1}/{num_runs}, seed={seed} ---")

            f = open('/data/usrs/yyr/cc/pricing/image_classification' + "/" + str(budget), 'r')
            lines = f.readlines()
            winners = [int(v) for v in lines[0].split()]
            trainloader, testloader = get_dataloader(trainset, testset, batch_size=128, indices = winners + list(init_indices))

            model = ResNet56().to(device)

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

            subset_indices = trainloader.dataset.indices
            X_subset = X_all[subset_indices]

            score = coverage_score(X_all, X_subset, 'image_classification')
            # score = summarization_score(X_all, X_subset)

            print(f"Run {run}: acc={test_acc:.2f}, score={score:.4f}")

            acc_list.append(test_acc)
            score_list.append(score)

        # ===== mean over runs =====
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)

        mean_score = np.mean(score_list)
        std_score = np.std(score_list)

        all_results.append(mean_acc)
        all_scores.append(mean_score)

        print(f"\n>>> Budget {budget} FINAL")
        print(f"Accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
        print(f"Score:    {mean_score:.4f} ± {std_score:.4f}")

    # =========================
    # final report
    # =========================
    print("\n==== FINAL SUMMARY ====")

    for budget, acc, score in zip(budgets, all_results, all_scores):
        print(f"{budget}: acc={acc:.2f}, score={score:.4f}")


if __name__ == "__main__":
    main()