from nats_bench import create

import torch
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import pickle
from p_tqdm import p_map

from torch_geometric.data import Data as PyG_Data


def get_config_data(args):
    api, dataset, idx = args
    config = api.get_net_config(idx, dataset)
    matrix = api.str2matrix(config['arch_str'])
    encoding = np.array(matrix[np.tril_indices(4, k=-1)], dtype=int)
    c_info = api.get_cost_info(idx, dataset)
    flops_, params_, latency_ = c_info['flops'], c_info['params'], c_info['latency']
    return encoding, flops_, params_, latency_


def get_pyg_data_list(input_data, target_data_lst=None):
    def transform_edge_index(v, ei):
        nei = torch.clone(ei)
        cols = set(range(10))
        if v[0] == 0.0:
            cols = cols - {0, 3, 4, 6, 7}
        if v[1] == 0.0:
            cols = cols - {1, 5}
        if v[2] == 0.0:
            cols = cols - {4, 6}
        if v[3] == 0.0:
            cols = cols - {2, 9}
        if v[4] == 0.0:
            cols = cols - {3, 7}
        if v[5] == 0.0:
            cols = cols - {1, 4, 5, 6, 8}
        return nei[:, list(cols)]

    # Constructing graphs
    #                           0  1  2  3  4  5  6  7  8  9
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 3, 5, 6, 4],
                               [1, 2, 4, 5, 3, 6, 6, 7, 7, 7]], dtype=torch.long)
    # x_ = [input, op1, op2, op3, op4, op5, op6, output]
    data_list = []
    if target_data_lst is None:
        for v in input_data:
            vector = torch.Tensor([0.0] + v.tolist() + [0.0]).unsqueeze(1)
            data = PyG_Data(x=vector, edge_index=transform_edge_index(v, edge_index), vector=v.unsqueeze(0))
            data_list.append(data)
    else:
        for idx, v in enumerate(input_data):
            vector = torch.Tensor([0.0] + v.tolist() + [0.0]).unsqueeze(1)
            y = [target_data[idx] for target_data in target_data_lst]
            data = PyG_Data(x=vector, edge_index=transform_edge_index(v, edge_index), y=y, vector=v.unsqueeze(0))
            data_list.append(data)

    return data_list


def generate_pretraining_data_file(filename, dataset, n_archs=2000, n_cpus=6):
    print(f'Search space:\tTSS\nDataset:\t{dataset}')
    api = create('../nasbench_201/data/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
    total_archs = 15625
    indices = np.random.default_rng().choice(total_archs, size=n_archs, replace=False)
    r = p_map(get_config_data,
              [(api, dataset, i) for i in indices],
              num_cpus=n_cpus)
    encodings, flops, params, latency = [], [], [], []
    for e in r:
        encodings.append(e[0])
        flops.append(e[1])
        params.append(e[2])
        latency.append(e[3])
    f = lambda x: torch.Tensor(np.array(x) / np.max(x))
    metrics = list(map(f, [flops, params, latency]))
    encodings = torch.Tensor(np.array(encodings))

    pyg_data_list = get_pyg_data_list(encodings, metrics)

    with open(filename, 'wb') as f_:
        pickle.dump(pyg_data_list, f_)


def get_acc_data(args):
    api, dataset, i = args
    config = api.get_net_config(i, dataset)
    matrix = api.str2matrix(config['arch_str'])
    encoding = np.array(matrix[np.tril_indices(4, k=-1)], dtype=int)
    d = set()
    for _ in range(10):
        d.add(api.get_more_info(i, dataset, hp='200')['test-accuracy'])
    acc = sum(d) / len(d)
    return encoding, acc


def generate_bench_data_file(filename, dataset, n_cpus=6):
    print(f'Search space:\tTSS\nDataset:\t{dataset}')
    api = create('../nasbench_201/data/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
    total_archs = 15625
    r = p_map(get_acc_data,
              [(api, dataset, i) for i in range(total_archs)],
              num_cpus=n_cpus)
    encodings, accuracies = [], []
    for e in r:
        encodings.append(e[0])
        accuracies.append(e[1])
    encodings = torch.Tensor(np.array(encodings))
    accuracies = torch.Tensor(np.array(accuracies) / 100)
    print(f'max={accuracies.max()}\tmin={accuracies.min()}\tmean={accuracies.mean()}\tstd={accuracies.std()}')

    pyg_data_list = get_pyg_data_list(encodings, [accuracies])

    with open(filename, 'wb') as f_:
        pickle.dump(pyg_data_list, f_)


if __name__ == '__main__':
    #datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
    datasets = ['ImageNet16-120']

    for d in datasets:
        filename_ = f'pretraining_data/nats_tss_{d}_metrics.pickle'
        generate_pretraining_data_file(filename_, d)

    for d in datasets:
        filename_ = f'pretraining_data/nats_tss_{d}_evals.pickle'
        generate_bench_data_file(filename_, d)
