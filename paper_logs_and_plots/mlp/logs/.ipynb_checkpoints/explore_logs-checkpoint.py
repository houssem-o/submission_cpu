import pickle
import numpy as np
import matplotlib.pyplot as plt
from os import mkdir
from os.path import exists


def load_logs(xp_list):
    logs = []
    for xp in xp_list:
        with open(xp, 'rb') as f:
            logs.append(pickle.load(f))
    return logs


def get_values_at_x_one_experiment(logs, n_evals):
    for i, log in enumerate(logs):
        if i + 1 < len(logs) and logs[i + 1]['n_evals'] > n_evals:
            return log['current_best'], log['n_evals']


def get_values_at_x(logs_list, n_evals):
    values = [get_values_at_x_one_experiment(l, n_evals)[0] for l in logs_list]
    n_evals = get_values_at_x_one_experiment(logs_list[0], n_evals)[1]
    return n_evals, np.mean(values)

def get_average_index_optimum_reached(lst):
    indices = [l[-1]['found_optimum_index'] for l in lst]
    clean_list = [i for i in indices if i is not None]
    n_fails = indices.count(None)
    if len(clean_list) == 0:
        return None
    return np.mean(clean_list), n_fails

def grab_data(lst):
    acc = np.array([l[-1]['current_best'] for l in lst]).mean()
    rho = np.array([l[-1]['correlations'][0].correlation for l in lst]).mean()
    idx, fs = get_average_index_optimum_reached(lst)
    return acc, rho, idx, fs

def get_index_stats(lst):
    x = [l[-1]['found_optimum_index'] for l in lst]
    return np.min(x), np.max(x), np.mean(x), np.std(x)

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
    dpi = 300

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
