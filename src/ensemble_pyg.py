import torch

torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyG_DataLoader
from joblib import Parallel, delayed
from tqdm import tqdm



class MultiHeadModulePyG(nn.Module):
    def __init__(self, encoder, heads_list):
        super(MultiHeadModulePyG, self).__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(heads_list)

    def forward(self, x, edge_index, batch_index, vector):
        arch_encoding = self.encoder(x, edge_index, batch_index, vector)
        preds = [head(arch_encoding) for head in self.heads]
        return preds


class PredictionHeadPyG(nn.Module):
    def __init__(self, net, embedding_dim):
        super(PredictionHeadPyG, self).__init__()
        self.net = net
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x, edge_index, batch_index, vector):
        return self.head(self.net(x, edge_index, batch_index, vector))


class EnsemblePyG():
    def __init__(self, pretrain_metrics_pyg_list, n_pretrain_metrics,
                 network_generator_func, embedding_dim,
                 n_networks=10, accelerator='cpu', devices=1, train_lr=5e-3,
                 pretrain_epochs=60, pretrain_lr=1e-3, pretrain_bs=16):
        self.embedding_dim = embedding_dim
        self.accelerator = accelerator
        self.devices = devices
        self.pretrain_epochs, self.pretrain_lr, self.pretrain_bs = pretrain_epochs, pretrain_lr, pretrain_bs

        self.networks = [
            PredictionHeadPyG(network_generator_func(), self.embedding_dim)
            for _ in range(n_networks)
        ]
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=train_lr) for net in self.networks]
        self.train_lr = train_lr

        self.pretrain_metrics_pyg_list = pretrain_metrics_pyg_list
        self.pretrain_modules = [MultiHeadModulePyG(encoder=net.net,
                                                    heads_list=[nn.Linear(self.embedding_dim, 1)
                                                                for _ in range(n_pretrain_metrics)])
                                 for net in self.networks]
        self.pretrain_optimizers = [torch.optim.Adam(module.parameters(), lr=self.pretrain_lr) for module in
                                    self.pretrain_modules]

    def pretrain(self):
        train_loader = PyG_DataLoader(self.pretrain_metrics_pyg_list, batch_size=self.pretrain_bs, shuffle=True, num_workers=0)

        def pretrain_epoch(module, optimizer):
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                preds = module(batch.x.float(), batch.edge_index, batch.batch, batch.vector)
                loss = 0
                # Preds: list of n_pretrain_metrics predictions
                for obj_pred, obj_target in zip(preds, batch.y):
                    loss += nn.functional.huber_loss(input=obj_pred, target=obj_target.view(obj_pred.shape))
                loss.backward()
                optimizer.step()
            return module, optimizer

        with Parallel(n_jobs=self.devices) as parallel:
            for _ in tqdm(range(self.pretrain_epochs)):
                res = parallel(
                    delayed(pretrain_epoch)(
                        module,
                        optimizer
                    )
                    for module, optimizer in zip(self.pretrain_modules, self.pretrain_optimizers)
                )
                self.pretrain_modules, self.pretrain_optimizers = [], []
                for (module, optimizer) in res:
                    self.pretrain_modules.append(module)
                    self.pretrain_optimizers.append(optimizer)

        # Restore self.networks and self.optimizers
        self.networks = [
            PredictionHeadPyG(module.get_submodule('encoder'), self.embedding_dim)
            for module in self.pretrain_modules
        ]
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=self.pretrain_lr) for net in self.networks]

    def train(self, pyg_data_list, epochs, bs=16):
        train_loader = PyG_DataLoader(pyg_data_list, shuffle=True, batch_size=bs)

        def train_epoch(net, optimizer):
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                preds = net(batch.x.float(), batch.edge_index, batch.batch, batch.vector)
                loss = nn.functional.huber_loss(input=preds, target=batch.y[0].view(preds.shape))
                loss.backward()
                optimizer.step()
            return net, optimizer

        with Parallel(n_jobs=self.devices) as parallel:
            for _ in range(epochs):
                res = parallel(
                    delayed(train_epoch)(
                        net,
                        optimizer
                    )
                    for net, optimizer in zip(self.networks, self.optimizers)
                )
                self.networks, self.optimizers = [], []
                for (net, optimizer) in res:
                    self.networks.append(net)
                    self.optimizers.append(optimizer)

