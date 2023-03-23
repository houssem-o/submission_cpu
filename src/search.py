import pickle
from scipy import stats
from os import mkdir
from os.path import exists
import torch
from tqdm import tqdm

torch.manual_seed(0)


class BenchSearch:
    def __init__(self, experiment_name, ensemble,
                 all_configs, all_evals,
                 is_optimum,
                 evals_per_iter=4, epochs_per_iter=2):
        self.ensemble = ensemble
        self.evals_per_iter = evals_per_iter
        self.epochs_per_iter = epochs_per_iter
        self.experiment_name = experiment_name

        self.logs = []
        self.best_so_far = torch.Tensor((0.0,))
        self.best_config = None

        self.all_configs = all_configs
        self.all_evals = all_evals
        self.mask = torch.zeros(all_configs.shape[0], dtype=int)

        self.found_optimum_index = None
        self.is_optimum = is_optimum

        if not (exists('logs')):
            mkdir('logs')

    def run(self, iterations):
        # BO Loop
        for _ in tqdm(range(iterations)):
            # Suggest next points to evaluate, based on acquisition function maximisation
            # Update dataset with new observations (simply update self.mask)
            self.max_acq_function()
            self.best_so_far = self.all_evals[self.mask==1].max()
            self.best_config = self.all_configs[self.all_evals[self.mask==1].argmax()]

            #print(self.mask.sum())
            # Update model
            self.ensemble.train(self.all_configs[self.mask==1],
                                self.all_evals[self.mask==1],
                                epochs=self.epochs_per_iter)

            # Log
            self.logs.append(self.evaluate_current_performance())

        with open(f'logs/{self.experiment_name}_logs.pickle', 'wb') as f:
            pickle.dump(self.logs, f)

        return self.logs

    def max_acq_function(self):
        tau = 1e-3
        configs = torch.Tensor(self.all_configs)
        preds = torch.stack([net(configs).squeeze() for net in self.ensemble.networks])
        impr = preds - self.best_so_far.to(preds)
        qPI = torch.sigmoid(impr / tau).mean(dim=0)
        n_pts = self.evals_per_iter
        for idx in torch.argsort(qPI, descending=True):
            if self.mask[idx] == 0:
                self.mask[idx] = 1
                n_pts -= 1
            if n_pts == 0:
                break

    def evaluate_current_performance(self):
        n_evals = int(self.mask.sum())
        current_best = self.best_so_far
        if self.found_optimum_index is None and self.is_optimum(current_best):
            self.found_optimum_index = n_evals
        cfgs, evals = self.all_configs, self.all_evals
        pred, std = self.ensemble.predict(cfgs)
        spearman_rho = stats.spearmanr(evals, pred)
        kendall_tau = stats.kendalltau(evals, pred)
        pearson_r = stats.pearsonr(evals, pred)
        return {'found_optimum_index': self.found_optimum_index,
                'n_evals': n_evals,
                'current_best': current_best,
                'current_best_config': self.best_config,
                'correlations': (spearman_rho, kendall_tau, pearson_r)}
