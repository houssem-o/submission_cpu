import torch
torch.manual_seed(0)
import torch.nn as nn
import numpy as np
np.random.seed(0)
import pickle
import argparse
from os import mkdir
from os.path import exists

from src.ensemble import Ensemble
from src.search import BenchSearch
from src.explore_logs import *


embedding_dim = 512


def encoder_generator_func_():
    return nn.Sequential(nn.Linear(6, 64), nn.LayerNorm(64),
                         nn.SiLU(),
                         nn.Linear(64, 128), nn.LayerNorm(128),
                         nn.SiLU(),
                         nn.Linear(128, 256), nn.LayerNorm(256),
                         nn.SiLU(),
                         nn.Linear(256, embedding_dim), nn.LayerNorm(512),
                         nn.SiLU())


def one_run(accel, threads, run_name, evals_filename, metrics_filename, bench_optimum, pretrain=True, plot_logs=True):
    # Ensemble instance and pretraining
    with open(metrics_filename, 'rb') as f:
        pretrain_configs, pretrain_metrics = pickle.load(f)
    e = Ensemble(pretrain_configs=pretrain_configs, pretrain_metrics=pretrain_metrics,
                 network_generator_func=encoder_generator_func_, embedding_dim=embedding_dim,
                 n_networks=6, accelerator=accel, devices=threads, train_lr=5e-3,
                 pretrain_epochs=1000, pretrain_lr=2e-3, pretrain_bs=2)
    if pretrain:
        e.pretrain()

    # Search
    with open(evals_filename, 'rb') as f:
        configs, accuracies = pickle.load(f)

    s = BenchSearch(experiment_name=run_name,
                    ensemble=e,
                    all_configs=configs,
                    all_evals=accuracies,
                    is_optimum=lambda x: bool(x >= bench_optimum),
                    evals_per_iter=32,
                    epochs_per_iter=30)

    logs = s.run(int(1024 / 32))

    if plot_logs:
        s.plot_logs()

    print(logs[-1]['found_optimum_index'])

    return logs


def multiple_runs(accel, threads, xp_name, evals_filename, metrics_filename, bench_optimum, pretrain=True, n_runs=10):
    logs_list = []
    for i in range(n_runs):
        logs_list.append(one_run(accel=accel, threads=threads,
                                 run_name=f'{xp_name}_{i}', evals_filename=evals_filename,
                                 metrics_filename=metrics_filename,
                                 bench_optimum=bench_optimum,
                                 pretrain=pretrain, plot_logs=False))
    return logs_list


def get_average_final_corr(lst):
    return np.array([l[-1]['correlations'][0].correlation for l in lst]).mean()


def get_average_index_optimum_reached(lst):
    indices = [l[-1]['found_optimum_index'] for l in lst]
    clean_list = [i for i in indices if i is not None]
    n_fails = indices.count(None)
    if len(clean_list) == 0:
        return None
    return np.mean(clean_list), n_fails

def create_dir_if_not_existing(path):
    if not exists(path):
        mkdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', type=str, default='nb201_pretrained')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--pretraining', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--datasets', type=str, nargs='+', default=['cifar10', 'cifar100', 'ImageNet16-120'])
    args = parser.parse_args()

    create_dir_if_not_existing('experiments/nasbench_201/plots')
    create_dir_if_not_existing('experiments/nasbench_201/logs')

    accel, threads, xp_name = args.accelerator, args.threads, args.experiment_name
    n_runs, pretrain, datasets = args.runs, args.pretraining, args.datasets

    bench_optima = {'cifar10': 0.9437, 'cifar100': 0.7351, 'ImageNet16-120': 0.4731}

    logs = dict()
    for d in datasets:
        evals_filename = f'experiments/nasbench_201/pretraining_data/nats_tss_{d}_evals.pickle'
        metrics_filename = f'experiments/nasbench_201/pretraining_data/nats_tss_{d}_metrics.pickle'
        logs[d] = multiple_runs(accel=accel, threads=threads,
                                xp_name=xp_name, evals_filename=evals_filename,
                                metrics_filename=metrics_filename,
                                bench_optimum=bench_optima[d],
                                pretrain=pretrain, n_runs=n_runs)

    snapshots = [100, 200, 400, 444]
    print('\nSummary')
    for d in datasets:
        print(f'{d}\t{n_runs} runs, {"With pretraining" if pretrain else "No pretraining"}')
        print('Acc = {}\tRho = {}\tIdx = {}\tF = {}'.format(*grab_data(logs[d])))
        print('Evals at optimum found: \tmin = {}\tmax = {}\tavg = {}\tstd = {}'.format(*get_index_stats(logs[d])))
        for s in snapshots:
            print('Snapshot at {}\t\tn_evals = {}\t\tacc = {}'.format(s, *get_values_at_x(logs[d], s)))
        print('\n')


# Code used for generating the experiment logs
# if __name__ == '__main__':
#     n_runs_no_pretraining, n_runs_pretrained = 10, 10

#     datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
#     bench_optima = {'cifar10': 0.9437, 'cifar100': 0.7351, 'ImageNet16-120': 0.4731}
#     best_val_no_pretraining = {}
#     best_val_pretraining = {}
#     last_spearman_coeff_no_pretraining = {}
#     last_spearman_coeff_pretraining = {}
#     avg_idx_no_pretraining = {}
#     avg_idx_pretraining = {}
#     for d in datasets:
#         xp_name_0 = f'nb201_{d}_no_pretraining'
#         evals_filename_0 = f'experiments/nasbench_201/pretraining_data/nats_tss_{d}_evals.pickle'
#         metrics_filename_0 = f'experiments/nasbench_201/pretraining_data/nats_tss_{d}_metrics.pickle'
#         logs_list_0 = multiple_runs(xp_name=xp_name_0, evals_filename=evals_filename_0,
#                                     metrics_filename=metrics_filename_0,
#                                     bench_optimum=bench_optima[d],
#                                     pretrain=False, n_runs=n_runs_no_pretraining)
#         best_val_no_pretraining[d] = np.array([l[-1]['current_best'] for l in logs_list_0]).mean()
#         last_spearman_coeff_no_pretraining[d] = np.array(
#             [l[-1]['correlations'][0].correlation for l in logs_list_0]).mean()
#         avg_idx_no_pretraining[d] = get_average_index_optimum_reached(logs_list_0)

#         xp_name_1 = f'nb201_{d}_pretrained'
#         evals_filename_1 = f'pretraining_data/nats_tss_{d}_evals.pickle'
#         metrics_filename_1 = f'pretraining_data/nats_tss_{d}_metrics.pickle'
#         logs_list_1 = multiple_runs(xp_name=xp_name_1, evals_filename=evals_filename_1,
#                                     metrics_filename=metrics_filename_1,
#                                     bench_optimum=bench_optima[d],
#                                     pretrain=True, n_runs=n_runs_pretrained)
#         best_val_pretraining[d] = np.array([l[-1]['current_best'] for l in logs_list_1]).mean()
#         last_spearman_coeff_pretraining[d] = np.array(
#             [l[-1]['correlations'][0].correlation for l in logs_list_1]).mean()

#         avg_idx_pretraining[d] = get_average_index_optimum_reached(logs_list_1)

#         plot_average_logs_multiple_experiments(d,
#                                                [logs_list_0, logs_list_1],
#                                                [xp_name_0, xp_name_1], True)

#     summary = ""
#     summary += f'NASBench-201 summary\n'
#     summary += '-' * 20 + '\n'
#     summary += f'No pretraining - {n_runs_no_pretraining} runs\n'
#     for d in datasets:
#         summary += f'Dataset:\t{d}\n'
#         summary += f'Best val (avg):\t{best_val_no_pretraining[d]}\n'
#         summary += f'Spearman corr. (avg):\t{last_spearman_coeff_no_pretraining[d]}\n'
#         summary += f'(Average index optimum reached, n_fails):\t{avg_idx_no_pretraining[d]}\n'
#     summary += '-' * 20 + '\n'
#     summary += f'With pretraining - {n_runs_pretrained} runs\n'
#     for d in datasets:
#         summary += f'Dataset:\t{d}\n'
#         summary += f'Best val (avg):\t{best_val_pretraining[d]}\n'
#         summary += f'Spearman corr. (avg):\t{last_spearman_coeff_pretraining[d]}\n'
#         summary += f'(Average index optimum reached, n_fails):\t{avg_idx_pretraining[d]}\n'
#     print(summary)
#     with open("summary.txt", "w") as f:
#         f.write(summary)
