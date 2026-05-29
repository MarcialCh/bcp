# Auction-based Data Pricing: Offline and Online

This repository implements the algorithms proposed in the paper *"Auction-based Data Pricing: Offline and Online"* together with their corresponding baselines.

## Implemented Algorithms

### Offline Mechanisms

The following offline algorithms are implemented:

* `bcp.py`
* `greedy.py`
* `lazy_threshold_bcp.py`

### Online Mechanisms

The following online algorithms are implemented:

* `post.py`
* `bid.py`
* `ms.py`

---

# Usage

## 1. Offline Experiments

### Crowdsensing Tasks

Run:

* `crowdsensing_hadian.py`
* `crowdsensing_lixia.py`

Before execution, set:

```
data_path = "datasets/haidianF_data.csv"
```

or

```python
data_path = "datasets/lixiaF_data.csv"
```

Also specify:

```
path
```

as the output directory for experimental results.

### Image Classification Tasks

Run:

* `train_CIFAR-10.py`

and set:

```
path
```

for result output.

To evaluate the trained model performance, run:

* `evaluate_CIFAR-10.py`

---

## 2. Online Experiments

### Crowdsensing Tasks

Run:

* `online_crowdsensing_hadian.py`
* `online_crowdsensing_lixia.py`

Before execution, set:

```python
data_path = "datasets/haidianF_data.csv"
```

or

```
data_path = "datasets/lixiaF_data.csv"
```

Also specify:

```python
path
```

as the output directory for experimental results.

### Image Classification Tasks

Run:

* `online_train_CIFAR-10.py`

and set:

```python
path
```

for result output.

To evaluate the trained model performance, run:

* `evaluate_CIFAR-10.py`
