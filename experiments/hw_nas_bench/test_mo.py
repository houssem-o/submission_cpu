import torch

torch.manual_seed(0)
import torch.nn as nn
import numpy as np

np.random.seed(0)
import pickle
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

from src.ensemble import EnsembleMO

embedding_dim = 512


class EncoderModule(nn.Module):
    def __init__(self, n_objectives):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(6, 64), nn.LayerNorm(64),
                                    nn.SiLU())
        self.specialized_list = nn.ModuleList([nn.Sequential(nn.Linear(64, 128), nn.LayerNorm(128),
                                                             nn.SiLU(),
                                                             nn.Linear(128, 256), nn.LayerNorm(256),
                                                             nn.SiLU(),
                                                             nn.Linear(256, embedding_dim), nn.LayerNorm(512),
                                                             nn.SiLU()) for _ in range(n_objectives)])
        self.embedding_dim = embedding_dim


def network_generator_func(n_objectives):
    return EncoderModule(n_objectives)


def run_training_once(train_configs, train_data, test_configs, test_data_list, metrics_filename, pretrain=True):
    with open(metrics_filename, 'rb') as f:
        pretrain_configs, pretrain_metrics = pickle.load(f)
    e = EnsembleMO(pretrain_configs=pretrain_configs, pretrain_metrics=pretrain_metrics,
                   network_generator_func=lambda: network_generator_func(n_objectives=len(test_data_list)),
                   embedding_dim=embedding_dim, n_objectives=len(test_data_list),
                   n_networks=6, accelerator='cpu', devices=6, train_lr=5e-3,
                   pretrain_epochs=100, pretrain_lr=2e-3, pretrain_bs=2)
    if pretrain:
        e.pretrain()

    total_evals_ = len(train_configs)
    n_evals_per_iter = 32
    epochs_per_iter = 30

    n_evals = []
    correlations = [[] for _ in range(len(test_data_list))]

    for i in tqdm(range(n_evals_per_iter, total_evals_ + 1, n_evals_per_iter)):
        # train on train_configs -> train_data
        e.train_multiple(input_data=train_configs[:i],
                         target_data=[e[:i] for e in train_data],
                         epochs=epochs_per_iter)
        # calculate spearman rank correlation
        predictions_list = e.predict(test_configs)
        for obj in range(len(test_data_list)):
            preds = predictions_list[obj]
            real = test_data_list[obj]
            correlations[obj].append(stats.spearmanr(preds, real).correlation)

        n_evals.append(i)

    return n_evals, correlations


def run_training(n_runs, train_configs, train_data, test_configs, test_data_list, metrics_filename, pretrain=True):
    results = [run_training_once(train_configs, train_data, test_configs, test_data_list, metrics_filename, pretrain)
               for _ in range(n_runs)]
    n_evals = results[0][0]
    avg_correlations = np.array([r[1] for r in results]).mean(axis=0)
    return n_evals, avg_correlations


def get_correlations(n_runs, configs, objectives, objective_names, evals, metrics_filename, pretrain, shared_only=False):
    d = dict()


    if not shared_only:
        # Separate
        for obj, obj_name in zip(objectives, objective_names):
            train_configs = configs[:evals]
            train_data = [obj[:evals]]

            test_configs = configs[evals:]
            test_data = [obj[evals:]]

            n_evals, correlations = run_training(n_runs=n_runs,
                                                 train_configs=train_configs,
                                                 train_data=train_data,
                                                 test_configs=test_configs,
                                                 test_data_list=test_data,
                                                 metrics_filename=metrics_filename,
                                                 pretrain=pretrain)

            d[obj_name + '_separate'] = (n_evals, correlations[0])

    # Shared
    train_configs = configs[:evals]
    train_data = [o[:evals] for o in objectives]

    test_configs = configs[evals:]
    test_data = [o[evals:] for o in objectives]

    n_evals, correlations = run_training(n_runs=n_runs,
                                         train_configs=train_configs,
                                         train_data=train_data,
                                         test_configs=test_configs,
                                         test_data_list=test_data,
                                         metrics_filename=metrics_filename,
                                         pretrain=pretrain)
    for obj_corr, obj_name in zip(correlations, objective_names):
        d[obj_name + '_shared'] = (n_evals, obj_corr[:])

    return d


def run_test(n_runs, configs, objectives, objective_names, evals, xp_name, metrics_filename, pretrain=True):
    d = get_correlations(n_runs, configs, objectives, objective_names, evals, metrics_filename, pretrain)

    for obj_name in objective_names:
        plt.figure(dpi=1200)
        #plt.title(f'Objective: {obj_name} - Spearman rank correlation')
        plt.plot(*d[obj_name + '_separate'], label=f'{obj_name} - Separate training')
        plt.plot(*d[obj_name + '_shared'], label=f'{obj_name} - Shared training')
        plt.legend(loc='lower right')
        plt.savefig(f'plots/{xp_name}_{obj_name}.png', bbox_inches='tight')


def compare_pretrained_trained(n_runs, configs, objectives, objective_names, evals, xp_name, metrics_filename):
    d_npr = get_correlations(n_runs, configs, objectives, objective_names, evals, metrics_filename, pretrain=False)
    d_pr = get_correlations(n_runs, configs, objectives, objective_names, evals, metrics_filename, pretrain=True)

    for obj_name in objective_names:
        plt.figure(dpi=1200)
        #plt.title(f'Objective: {obj_name} - Spearman rank correlation')
        plt.plot(*d_npr[obj_name + '_separate'], label=f'No pretraining - Separate training')
        plt.plot(*d_pr[obj_name + '_separate'], label=f'Pretrained - Separate training')
        plt.plot(*d_npr[obj_name + '_shared'], label=f'No pretraining - Shared training')
        plt.plot(*d_pr[obj_name + '_shared'], label=f'Pretrained - Shared training')
        plt.legend(loc='lower right')
        plt.savefig(f'plots/not_pretrained_vs_pretrained/{xp_name}_prVnpr_{obj_name}.png', bbox_inches='tight')


def compare_shared_training_2_experiments(n_runs, configs,
                                          objectives_1, objective_names_1, label_1,
                                          objectives_2, objective_names_2, label_2,
                                          evals, xp_name, metrics_filename, pretrain=True):
    d_1 = get_correlations(n_runs, configs, objectives_1, objective_names_1, evals, metrics_filename, pretrain)
    d_2 = get_correlations(n_runs, configs, objectives_2, objective_names_2, evals, metrics_filename, pretrain)
    common_objective_names = [o for o in objective_names_1 if o in objective_names_2]

    for obj_name in common_objective_names:
        plt.figure(dpi=1200)
        #plt.title(f'Objective: {obj_name} - Spearman rank correlation')
        plt.plot(*d_1[obj_name + '_shared'], label=label_1)
        plt.plot(*d_2[obj_name + '_shared'], label=label_2)
        plt.legend(loc='lower right')
        plt.savefig(f'plots/2v3_objectives/{xp_name}_{obj_name}.png', bbox_inches='tight')


if __name__ == '__main__':
    total_evals = 1024
    n_runs = 5

    evals_filename = "pretraining_data/hw_n201_cifar10_evals.pickle"
    metrics_filename = "pretraining_data/hw_n201_cifar10_metrics.pickle"

    with open(evals_filename, 'rb') as f:
        encodings, accuracies, \
        edgegpu_latencies, edgegpu_energies, \
        raspi4_latencies, edgetpu_latencies, \
        pixel3_latencies, eyeriss_latencies, \
        eyeriss_energies, eyeriss_arithmetic_intensities = pickle.load(f)

    # Plots
    with plt.style.context('ggplot'):
        # Eyeriss, 4 objectives
        run_test(n_runs=n_runs,
                 configs=encodings,
                 objectives=[accuracies, eyeriss_latencies, eyeriss_energies, eyeriss_arithmetic_intensities],
                 objective_names=['Accuracy', 'Latency', 'Energy', 'Arithmetic Intensity'],
                 evals=total_evals, xp_name='Eyeriss',
                 metrics_filename=metrics_filename, pretrain=True)


        # Pixel 3, no pretraining
        run_test(n_runs=n_runs,
                 configs=encodings,
                 objectives=[accuracies, pixel3_latencies],
                 objective_names=['Accuracy', 'Latency'],
                 evals=total_evals, xp_name='Pixel3',
                 metrics_filename=metrics_filename, pretrain=False)

        # EdgeTPU pretrained vs not pretrained
        compare_pretrained_trained(n_runs=n_runs,
                                   configs=encodings,
                                   objectives=[accuracies, edgetpu_latencies],
                                   objective_names=['Accuracy', 'Latency'],
                                   evals=total_evals, xp_name='EdgeTPU',
                                   metrics_filename=metrics_filename)

        # EdgeGPU 2 vs 3 objectives
        compare_shared_training_2_experiments(n_runs=n_runs,
                                              configs=encodings,
                                              objectives_1=[accuracies, edgegpu_energies],
                                              objective_names_1=['Accuracy', 'Energy'],
                                              label_1='2 objectives used',
                                              objectives_2=[accuracies, edgegpu_latencies, edgegpu_energies],
                                              objective_names_2=['Accuracy', 'Latency', 'Energy'],
                                              label_2='3 objectives used',
                                              evals=total_evals, xp_name='EdgeTPU',
                                              metrics_filename=metrics_filename,
                                              pretrain=True)
