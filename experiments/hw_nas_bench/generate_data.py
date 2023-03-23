from nats_bench import create
from benchmark.hw_nas_bench_api import HWNASBenchAPI as HWAPI

import torch
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import pickle
from p_tqdm import p_map


def get_config_data(args):
    api, dataset, idx = args
    config = api.get_net_config(idx, dataset)
    matrix = api.str2matrix(config['arch_str'])
    encoding = np.array(matrix[np.tril_indices(4, k=-1)], dtype=int)
    c_info = api.get_cost_info(idx, dataset)
    flops_, params_, latency_ = c_info['flops'], c_info['params'], c_info['latency']
    return encoding, flops_, params_, latency_


def generate_pretraining_data_file(filename, dataset, n_archs=2000, n_cpus=6):
    print(f'Search space:\tTSS\nDataset:\t{dataset}')
    api = create('../nasbench_201/data/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True,
                 verbose=False)
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
    with open(filename, 'wb') as f_:
        pickle.dump((encodings, metrics), f_)


def get_acc_data(args):
    api, hw_api, dataset, i, val = args
    config = api.get_net_config(i, dataset)
    matrix = api.str2matrix(config['arch_str'])
    encoding = np.array(matrix[np.tril_indices(4, k=-1)], dtype=int)
    if val:
        # returns validation_accuracy, latency, time_cost, current_total_time_cost
        acc, _, _, _ = api.simulate_train_eval(i, dataset=dataset, hp='200')
    else:
        d = set()
        for _ in range(10):
            d.add(api.get_more_info(i, dataset, hp='200')['test-accuracy'])
        acc = sum(d) / len(d)
    hw = hw_api.query_by_index(i, dataset)
    hw['encoding'] = encoding
    hw['accuracy'] = acc
    return hw


def generate_bench_data_file(filename, dataset, val=False, n_cpus=6):
    print(f'Search space:\tTSS\nDataset:\t{dataset}')
    api = create('../nasbench_201/data/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True,
                 verbose=False)
    hw_api = HWAPI("benchmark/HW-NAS-Bench-v1_0.pickle", search_space='nasbench201')
    total_archs = 15625
    r = p_map(get_acc_data,
              [(api, hw_api, dataset, i, val) for i in range(total_archs)],
              num_cpus=n_cpus)

    encodings, accuracies, edgegpu_latencies, edgegpu_energies, \
    raspi4_latencies, edgetpu_latencies, pixel3_latencies, \
    eyeriss_latencies, eyeriss_energies, eyeriss_arithmetic_intensities = [], [], [], [], [], [], [], [], [], []
    for e in r:
        encodings.append(e['encoding'])
        accuracies.append(e['accuracy'])
        edgegpu_latencies.append(e['edgegpu_latency'])
        edgegpu_energies.append(e['edgegpu_energy'])
        raspi4_latencies.append(e['raspi4_latency'])
        edgetpu_latencies.append(e['edgetpu_latency'])
        pixel3_latencies.append(e['pixel3_latency'])
        eyeriss_latencies.append(e['eyeriss_latency'])
        eyeriss_energies.append(e['eyeriss_energy'])
        eyeriss_arithmetic_intensities.append(e['eyeriss_arithmetic_intensity'])

    encodings = torch.Tensor(np.array(encodings))
    accuracies = torch.Tensor(np.array(accuracies) / 100)
    f = lambda x: torch.Tensor(np.array(x) / np.max(x))
    edgegpu_latencies, edgegpu_energies, \
    raspi4_latencies, edgetpu_latencies, \
    pixel3_latencies, eyeriss_latencies, \
    eyeriss_energies, eyeriss_arithmetic_intensities = map(f, [edgegpu_latencies, edgegpu_energies,
                                                               raspi4_latencies, edgetpu_latencies,
                                                               pixel3_latencies, eyeriss_latencies,
                                                               eyeriss_energies, eyeriss_arithmetic_intensities])

    with open(filename, 'wb') as f_:
        pickle.dump((encodings, accuracies,
                     edgegpu_latencies, edgegpu_energies,
                     raspi4_latencies, edgetpu_latencies,
                     pixel3_latencies, eyeriss_latencies,
                     eyeriss_energies, eyeriss_arithmetic_intensities), f_)


if __name__ == '__main__':
    datasets = ['cifar10']

    for d in datasets:
        filename_ = f'pretraining_data/hw_n201_{d}_metrics.pickle'
        generate_pretraining_data_file(filename_, d)

    for d in datasets:
        filename_ = f'pretraining_data/hw_n201_{d}_evals.pickle'
        generate_bench_data_file(filename_, d)
