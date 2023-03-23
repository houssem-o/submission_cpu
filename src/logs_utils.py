import matplotlib.pyplot as plt
import numpy as np
from os import mkdir
from os.path import exists


def plot_average_logs_one_experiment(logs_list, bench=False):
    current_best_avg, current_best_std = [], []
    spearman_rho_avg, spearman_rho_std = [], []
    kendall_tau_avg, kendall_tau_std = [], []
    pearson_r_avg, pearson_r_std = [], []
    for t in range(len(logs_list[0])):
        current_best_lst = [log[t]['current_best'] for log in logs_list]
        current_best_avg.append(np.mean(current_best_lst))
        current_best_std.append(np.std(current_best_lst))

        if bench:
            current_best_lst = [log[t]['correlations'][0].correlation for log in logs_list]
            spearman_rho_avg.append(np.mean(current_best_lst))
            spearman_rho_std.append(np.std(current_best_lst))

            kendall_tau_lst = [log[t]['correlations'][1][0] for log in logs_list]
            kendall_tau_avg.append(np.mean(kendall_tau_lst))
            kendall_tau_std.append(np.std(kendall_tau_lst))

            pearson_r_lst = [log[t]['correlations'][2].statistic for log in logs_list]
            pearson_r_avg.append(np.mean(pearson_r_lst))
            pearson_r_std.append(np.std(pearson_r_lst))

    n_evals = [logs_list[0][t]['n_evals'] for t in range(len(logs_list[0]))]

    return np.array(n_evals), \
           np.array(current_best_avg), np.array(current_best_std), \
           np.array(spearman_rho_avg), np.array(spearman_rho_std), \
           np.array(kendall_tau_avg), np.array(kendall_tau_std), \
           np.array(pearson_r_avg), np.array(pearson_r_std)


def plot_average_logs_multiple_experiments(prefix, logs_list_of_lists, names, bench=False, title=True):
    fill_between_coeff = 0.0
    dpi = 600

    res = []
    for logs_list in logs_list_of_lists:
        res.append(plot_average_logs_one_experiment(logs_list, bench))

    # Plotting best value
    plt.figure(dpi=dpi)
    if title:
        plt.title('Best value by number of evaluations')
    for lists, label in zip(res, names):
        plt.plot(lists[0], lists[1], label=label)
        plt.fill_between(lists[0],
                         lists[1] - fill_between_coeff * lists[2],
                         lists[1] + fill_between_coeff * lists[2], alpha=0.2)
    plt.legend(loc='lower right')
    plt.savefig(f'plots/{prefix}_best_val_comparison.png', bbox_inches='tight')

    if bench:
        plt.figure(dpi=dpi)
        if title:
            plt.title('Spearman rho rank correlation by number of evaluations')
        for lists, label in zip(res, names):
            plt.plot(lists[0], lists[3], label=label)
            plt.fill_between(lists[0],
                             lists[3] - fill_between_coeff * lists[4],
                             lists[3] + fill_between_coeff * lists[4], alpha=0.2)
        plt.legend(loc='lower right')
        plt.savefig(f'plots/{prefix}_spearman_rho_comparison.png', bbox_inches='tight')

        plt.figure(dpi=dpi)
        if title:
            plt.title('Kendall tau rank correlation by number of evaluations')
        for lists, label in zip(res, names):
            plt.plot(lists[0], lists[5], label=label)
            plt.fill_between(lists[0],
                             lists[5] - fill_between_coeff * lists[6],
                             lists[5] + fill_between_coeff * lists[6], alpha=0.2)
        plt.legend(loc='lower right')
        plt.savefig(f'plots/{prefix}_kendall_tau_comparison.png', bbox_inches='tight')

        plt.figure(dpi=dpi)
        if title:
            plt.title('Pearson R rank correlation by number of evaluations')
        for lists, label in zip(res, names):
            plt.plot(lists[0], lists[7], label=label)
            plt.fill_between(lists[0],
                             lists[7] - fill_between_coeff * lists[8],
                             lists[7] + fill_between_coeff * lists[8], alpha=0.2)
        plt.legend(loc='lower right')
        plt.savefig(f'plots/{prefix}_pearson_r_comparison.png', bbox_inches='tight')


def plot_log(log, experiment_name, bench=False):
    if not (exists('plots')):
        mkdir('plots')
    n_evals, best_values, spearman_rho, kendall_tau, pearson_r = [], [], [], [], []
    for l in log:
        n_evals.append(l['n_evals'])
        best_values.append(l['current_best'])
        if bench:
            spearman_rho.append(l['correlations'][0].correlation)
            kendall_tau.append(l['correlations'][1][0])
            pearson_r.append(l['correlations'][2].statistic)
    plots = [best_values]
    titles = ['Best value by number of evaluations']
    if bench:
        plots.extend([spearman_rho, kendall_tau, pearson_r])
        titles.extend(['Spearman rho rank correlation by number of evaluations',
                       'Kendall tau rank correlation by number of evaluations',
                       'Pearson R rank correlation by number of evaluations'])
    for y, title in zip(plots, titles):
        plt.figure()
        plt.title(title)
        plt.plot(n_evals, y)
        plt.savefig(f'plots/{experiment_name}_{title}.png', bbox_inches='tight')
