import torch

torch.manual_seed(0)
import torch.nn as nn
import numpy as np

np.random.seed(0)
import pickle
from torch_geometric.nn import GCNConv, aggr

from src.ensemble_pyg import EnsemblePyG
from src.search_pyg import BenchSearchPyG
from src.logs_utils import plot_average_logs_multiple_experiments

embedding_dim = 512


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)
        self.agg = aggr.MLPAggregation(256, 32, 6, num_layers=1)
        self.v = nn.Linear(6, 256)
        self.classifier = nn.Sequential(nn.LayerNorm(32 + 256), nn.SiLU(),
                                        nn.Linear(32 + 256, 512), nn.LayerNorm(512),
                                        nn.SiLU(),
                                        nn.Linear(512, 512), nn.LayerNorm(512),
                                        nn.SiLU())

    def forward(self, x, edge_index, batch_index, vector):
        h_ = self.conv1(x, edge_index)
        h_ = h_.relu()
        h_ = self.conv2(h_, edge_index)
        h_ = h_.relu()
        h_ = self.conv3(h_, edge_index)
        h_ = h_.relu()
        h_ = self.agg(h_, batch_index)
        h_ = torch.cat([h_, self.v(vector)], dim=1)
        out = self.classifier(h_)
        return out


def encoder_generator_func_():
    return GCN()


def one_run(run_name, evals_filename, metrics_filename, bench_optimum, pretrain=True, plot_logs=True):
    # Ensemble instance and pretraining
    with open(metrics_filename, 'rb') as f:
        pretrain_metrics_pyg_list = pickle.load(f)

    e = EnsemblePyG(pretrain_metrics_pyg_list=pretrain_metrics_pyg_list, n_pretrain_metrics=3,
                    network_generator_func=encoder_generator_func_, embedding_dim=embedding_dim,
                    n_networks=6, accelerator='cpu', devices=6, train_lr=5e-3,
                    pretrain_epochs=1000, pretrain_lr=5e-3, pretrain_bs=8)

    if pretrain:
        e.pretrain()

    # Search
    with open(evals_filename, 'rb') as f:
        all_pyg_data_list = pickle.load(f)

    s = BenchSearchPyG(experiment_name=run_name,
                       ensemble=e,
                       all_pyg_data_list=all_pyg_data_list,
                       is_optimum=lambda x: bool(x >= bench_optimum),
                       evals_per_iter=32,
                       epochs_per_iter=100)

    #logs = s.run(int(1024 / 32))
    logs = s.run(int(32*15 / 32))

    if plot_logs:
        s.plot_logs()

    print(logs[-1]['found_optimum_index'])

    return logs


def multiple_runs(xp_name, evals_filename, metrics_filename, bench_optimum, pretrain=True, n_runs=10):
    logs_list = []
    for i in range(n_runs):
        logs_list.append(one_run(run_name=f'{xp_name}_{i}', evals_filename=evals_filename,
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


if __name__ == '__main__':
    n_runs_no_pretraining, n_runs_pretrained = 1, 5

    #datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
    datasets = ['ImageNet16-120']

    bench_optima = {'cifar10': 0.9437, 'cifar100': 0.7351, 'ImageNet16-120': 0.4731}
    best_val_no_pretraining = {}
    best_val_pretraining = {}
    last_spearman_coeff_no_pretraining = {}
    last_spearman_coeff_pretraining = {}
    avg_idx_no_pretraining = {}
    avg_idx_pretraining = {}
    for d in datasets:
        # xp_name_0 = f'nb201_{d}_no_pretraining'
        # evals_filename_0 = f'pretraining_data/nats_tss_{d}_evals.pickle'
        # metrics_filename_0 = f'pretraining_data/nats_tss_{d}_metrics.pickle'
        # logs_list_0 = multiple_runs(xp_name=xp_name_0, evals_filename=evals_filename_0,
        #                             metrics_filename=metrics_filename_0,
        #                             bench_optimum=bench_optima[d],
        #                             pretrain=False, n_runs=n_runs_no_pretraining)
        # best_val_no_pretraining[d] = np.array([l[-1]['current_best'] for l in logs_list_0]).mean()
        # last_spearman_coeff_no_pretraining[d] = np.array(
        #     [l[-1]['correlations'][0].correlation for l in logs_list_0]).mean()
        # avg_idx_no_pretraining[d] = get_average_index_optimum_reached(logs_list_0)

        xp_name_1 = f'nb201_{d}_pretrained'
        evals_filename_1 = f'pretraining_data/nats_tss_{d}_evals.pickle'
        metrics_filename_1 = f'pretraining_data/nats_tss_{d}_metrics.pickle'
        logs_list_1 = multiple_runs(xp_name=xp_name_1, evals_filename=evals_filename_1,
                                    metrics_filename=metrics_filename_1,
                                    bench_optimum=bench_optima[d],
                                    pretrain=True, n_runs=n_runs_pretrained)
        best_val_pretraining[d] = np.array([l[-1]['current_best'] for l in logs_list_1]).mean()
        last_spearman_coeff_pretraining[d] = np.array(
            [l[-1]['correlations'][0].correlation for l in logs_list_1]).mean()

        avg_idx_pretraining[d] = get_average_index_optimum_reached(logs_list_1)

        # plot_average_logs_multiple_experiments(d,
        #                                        [logs_list_0, logs_list_1],
        #                                        [xp_name_0, xp_name_1], True)

    summary = ""
    summary += f'NASBench-201 summary\n'
    # summary += '-' * 20 + '\n'
    # summary += f'No pretraining - {n_runs_no_pretraining} runs\n'
    # for d in datasets:
    #     summary += f'Dataset:\t{d}\n'
    #     summary += f'Best val (avg):\t{best_val_no_pretraining[d]}\n'
    #     summary += f'Spearman corr. (avg):\t{last_spearman_coeff_no_pretraining[d]}\n'
    #     summary += f'(Average index optimum reached, n_fails):\t{avg_idx_no_pretraining[d]}\n'
    summary += '-' * 20 + '\n'
    summary += f'With pretraining - {n_runs_pretrained} runs\n'
    for d in datasets:
        summary += f'Dataset:\t{d}\n'
        summary += f'Best val (avg):\t{best_val_pretraining[d]}\n'
        summary += f'Spearman corr. (avg):\t{last_spearman_coeff_pretraining[d]}\n'
        summary += f'(Average index optimum reached, n_fails):\t{avg_idx_pretraining[d]}\n'
    print(summary)
    with open("summary.txt", "w") as f:
        f.write(summary)
