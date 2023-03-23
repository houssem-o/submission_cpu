import pickle
from itertools import compress
from scipy import stats
from os import mkdir
from os.path import exists
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyG_DataLoader

torch.manual_seed(0)


class BenchSearchPyG:
    def __init__(self, experiment_name, ensemble,
                 all_pyg_data_list,
                 is_optimum,
                 evals_per_iter=4, epochs_per_iter=2):
        self.ensemble = ensemble
        self.evals_per_iter = evals_per_iter
        self.epochs_per_iter = epochs_per_iter
        self.experiment_name = experiment_name

        self.logs = []
        self.best_so_far = torch.Tensor((0.0,))
        self.best_config = None

        self.all_pyg_data_list = all_pyg_data_list
        self.all_data_loader = PyG_DataLoader(all_pyg_data_list, batch_size=len(all_pyg_data_list),
                                              shuffle=False, num_workers=0)
        self.mask = [False] * len(all_pyg_data_list)

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
            evaluated_lst = list(compress(self.all_pyg_data_list, self.mask))
            evals = torch.Tensor([d.y[0] for d in evaluated_lst])
            self.best_so_far = evals.max()
            self.best_config = self.all_pyg_data_list[evals.argmax()].vector

            # print(self.mask.sum())
            # Update model
            self.ensemble.train(evaluated_lst,
                                epochs=self.epochs_per_iter)

            # Log
            self.logs.append(self.evaluate_current_performance())

        with open(f'logs/{self.experiment_name}_logs.pickle', 'wb') as f:
            pickle.dump(self.logs, f)

        return self.logs

    def max_acq_function(self):
        tau = 1e-3
        with torch.no_grad():
            preds = []
            for net in self.ensemble.networks:
                batch = next(iter(self.all_data_loader))
                preds.append(net(batch.x.float(), batch.edge_index, batch.batch, batch.vector).squeeze())
        preds = torch.stack(preds)
        impr = preds - self.best_so_far.to(preds)
        qPI = torch.sigmoid(impr / tau).mean(dim=0)
        n_pts = self.evals_per_iter
        for idx in torch.argsort(qPI, descending=True):
            if not self.mask[idx]:
                self.mask[idx] = True
                n_pts -= 1
            if n_pts == 0:
                break

    def evaluate_current_performance(self):
        n_evals = sum(self.mask)
        current_best = self.best_so_far
        if self.found_optimum_index is None and self.is_optimum(current_best):
            self.found_optimum_index = n_evals
        with torch.no_grad():
            preds, evals = [], []
            for net in self.ensemble.networks:
                batch = next(iter(self.all_data_loader))
                preds.append(net(batch.x.float(), batch.edge_index, batch.batch, batch.vector))
                evals = batch.y[0].cpu().numpy()
        preds = torch.stack(preds).mean(dim=0).squeeze().cpu().numpy()
        spearman_rho = stats.spearmanr(evals, preds)
        kendall_tau = stats.kendalltau(evals, preds)
        pearson_r = stats.pearsonr(evals, preds)
        return {'found_optimum_index': self.found_optimum_index,
                'n_evals': n_evals,
                'current_best': current_best,
                'current_best_config': self.best_config,
                'correlations': (spearman_rho, kendall_tau, pearson_r)}
