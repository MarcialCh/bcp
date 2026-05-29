import numpy as np
import pandas as pd
from cost import get_normalized_costs
from offline.bcp import greedy_allocation
import torch
import time

from offline.lazy_threshold_bcp import th_greedy_allocation
from online.ms import multi_stage_mechanism
from offline.greey import greedy_mechanism
from score import coverage_score

data_path = ''

path = ''

def load_poi_coordinates(csv_path):
    """
    return:
        X_all: torch.Tensor, shape [N, 2]
    """
      
    df = pd.read_csv(csv_path, encoding='utf-8')

    
    coords = df['location'].str.split('#', expand=True)

    
    longitude = coords[0].astype(float)
    latitude = coords[1].astype(float)

    
    X_all = torch.tensor(
        list(zip(longitude, latitude)),
        dtype=torch.float32
    )

    return X_all



def sample_by_region(X_all, center_idx, r):

    center = X_all[center_idx].to(X_all.device)

    diff = X_all - center

    dist2 = (diff ** 2).sum(dim=1)

    mask = dist2 <= r ** 2

    indices = torch.where(mask)[0]

    return X_all[indices], indices


def find_densest_center(X_all, r):

    """
    return:
        center_idx
        center
        neighbor_count
    """

    diff = X_all[:, None, :] - X_all[None, :, :]

    dist2 = np.sum(diff ** 2, axis=-1)

    neighbors = dist2 <= r**2

    counts = neighbors.sum(axis=1)

    center_idx = np.argmax(counts)

    center = X_all[center_idx]

    return center_idx, center, counts[center_idx]

# =========================
# main
# =========================
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    r = 4.5e-4

    s = 25

    torch.manual_seed(0)
    np.random.seed(0)
    
    X_all = load_poi_coordinates(data_path)

    # center_idx = np.random.randint(len(X_all))
    

    # center_idx, center, density = find_densest_center(X_all, r*s)

    center_idx = 0

    print(f'center_idx:{center_idx}')

    X_all, _ = sample_by_region(X_all, center_idx , r*s)

    print(f'size:{X_all.size(0)}')

    budgets = list(range(500, 501, 100))

    epsilon = 0.05

    task = 'offline_lz'

    # task = 'offline_bcp'

    # task = 'offline_greedy'

    init_indices =  np.random.choice(X_all.shape[0], size=100, replace=False)



    for budget in budgets:

        start = time.perf_counter()

        costs = get_normalized_costs(X_all, 'crowdsensing')

        if task == 'offline_bcp':

            winners, current_gain, trace = greedy_allocation(X_all=X_all, 
                                                             init_indices=list(init_indices), 
                                                             costs=costs, 
                                                             budget=budget, task_type='crowdsensing', r=r)

        if task == 'offline_lz':
            winners, current_gain, trace = th_greedy_allocation(X_all=X_all, 
                                                                init_indices=list(init_indices), 
                                                                costs=costs, budget=budget, epsilon=epsilon, task_type='crowdsensing', r=r)
            
        if task == 'offline_greedy':
            winners, current_gain, trace = greedy_mechanism(X_all,
                                                            init_indices=list(init_indices),
                                                            costs=costs,
                                                            budget=budget,
                                                            task_type = 'crowdsensing', r=r)


        end = time.perf_counter()

        running_time = end - start

        print(f"time: {running_time:.6f}s")

        write_file = open(path + "/" + str(task) + "_" + str(epsilon) + "_" + str(budget), 'w')
        write_file.write(' '.join([str(v) for v in winners]) + "\n")
        write_file.write(str(current_gain) + "\n")
        write_file.write(str(running_time) + "\n")
        # write_file.write(' '.join([str(u) for u in utility]) + "\n")
        write_file.flush()
        write_file.close()   



if __name__ == "__main__":
    main()



    
