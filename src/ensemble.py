import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed
from tqdm import tqdm


class MultiHeadModule(nn.Module):
    def __init__(self, encoder, heads_list):
        super(MultiHeadModule, self).__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(heads_list)

    def forward(self, configs):
        arch_encoding = self.encoder(configs)
        preds = [head(arch_encoding) for head in self.heads]
        return preds


class Ensemble:

    def __init__(self, pretrain_configs, pretrain_metrics,
                 network_generator_func, embedding_dim,
                 n_networks=10, accelerator='cpu', devices=1, train_lr=5e-3,
                 pretrain_epochs=60, pretrain_lr=1e-3, pretrain_bs=16):
        self.embedding_dim = embedding_dim
        self.accelerator = accelerator
        self.devices = devices
        self.pretrain_epochs, self.pretrain_lr, self.pretrain_bs = pretrain_epochs, pretrain_lr, pretrain_bs

        self.networks = [
            nn.Sequential(network_generator_func(), nn.Linear(self.embedding_dim, 1))
            for _ in range(n_networks)
        ]
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=train_lr) for net in self.networks]
        self.train_lr = train_lr

        self.pretrain_metrics = pretrain_metrics
        self.pretrain_configs = pretrain_configs
        self.pretrain_modules = [MultiHeadModule(encoder=net.get_submodule('0'),
                                                 heads_list=[nn.Linear(self.embedding_dim, 1)
                                                            for _ in range(len(self.pretrain_metrics))])
                                 for net in self.networks]
        self.pretrain_optimizers = [torch.optim.Adam(module.parameters(), lr=self.pretrain_lr) for module in
                                    self.pretrain_modules]

    def pretrain_cpu(self):
        train_set = TensorDataset(self.pretrain_configs, *self.pretrain_metrics)
        train_loader = DataLoader(train_set, batch_size=self.pretrain_bs, shuffle=True, num_workers=0)

        def pretrain_epoch(module, optimizer):
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                preds = module(batch[0])
                loss = 0
                for i in range(1, len(batch)):
                    loss += nn.functional.huber_loss(input=preds[i - 1], target=batch[i].view(preds[i - 1].shape))
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

    def pretrain_multi_gpu(self):
        train_set = TensorDataset(self.pretrain_configs, *self.pretrain_metrics)
        train_loader = DataLoader(train_set, batch_size=self.pretrain_bs, shuffle=True, num_workers=0)

        def pretrain_module(module, optimizer):
            device = torch.device(f'cuda:0')
            module.to(device)
            for _ in tqdm(range(self.pretrain_epochs)):
                for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    preds = module(batch[0].to(device))
                    loss = 0
                    for i in range(1, len(batch)):
                        loss += nn.functional.huber_loss(input=preds[i - 1].to(device), target=batch[i].to(device).view(preds[i - 1].shape))
                    loss.backward()
                    optimizer.step()
            return module.to(torch.device('cpu')), optimizer

        with Parallel(n_jobs=self.devices) as parallel:
            res = parallel(
                delayed(pretrain_module)(
                    module,
                    optimizer,
                    cuda_device
                    )
                )
                for module, optimizer in zip(self.pretrain_modules, self.pretrain_optimizers)


    def pretrain_gpu(self):
        train_set = TensorDataset(self.pretrain_configs, *self.pretrain_metrics)
        train_loader = DataLoader(train_set, batch_size=self.pretrain_bs, shuffle=True, num_workers=0)

        device = torch.device("cuda")

        for module, optimizer in zip(self.pretrain_modules, self.pretrain_optimizers):
            module.to(device)
            for _ in tqdm(range(self.pretrain_epochs)):
                for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    preds = module(batch[0].to(device))
                    loss = 0
                    for i in range(1, len(batch)):
                        loss += nn.functional.huber_loss(input=preds[i - 1].to(device), target=batch[i].to(device).view(preds[i - 1].shape))
                    loss.backward()
                    optimizer.step()
            module.to(torch.device('cpu'))


    def pretrain(self):
        if self.accelerator == 'cpu':
            self.pretrain_cpu()
        elif self.accelerator == 'gpu':
            self.pretrain_gpu()
        # Restore self.networks and self.optimizers
        self.networks = [
            nn.Sequential(module.get_submodule('encoder'), nn.Linear(self.embedding_dim, 1))
            for module in self.pretrain_modules
        ]
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=self.pretrain_lr) for net in self.networks]
        

    def train(self, input_data, target_data, epochs, bs=16):
        train_set = TensorDataset(input_data, target_data)
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)

        def train_epoch(net, optimizer):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                preds = net(inputs)
                loss = nn.functional.huber_loss(input=preds, target=targets.view(preds.shape))
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

    def predict(self, input_data):
        # input_data.to(torch.device(self.accelerator))
        with torch.no_grad():
            preds = torch.stack([net(input_data) for net in self.networks])
            return preds.mean(dim=0).detach().cpu().numpy(), preds.std(dim=0).detach().cpu().numpy()


class EnsembleMO(Ensemble):
    def __init__(self, pretrain_configs, pretrain_metrics,
                 network_generator_func,
                 embedding_dim, n_objectives,
                 n_networks=10, accelerator='cpu', devices=1, train_lr=5e-3,
                 pretrain_epochs=60, pretrain_lr=1e-3, pretrain_bs=16):
        self.accelerator = accelerator
        self.devices = devices
        self.pretrain_epochs, self.pretrain_lr, self.pretrain_bs = pretrain_epochs, pretrain_lr, pretrain_bs
        self.pretrain_metrics = pretrain_metrics
        self.pretrain_configs = pretrain_configs
        self.embedding_dim = embedding_dim
        self.n_objectives = n_objectives
        self.n_networks = n_networks

        self.encoders = [network_generator_func() for _ in range(n_networks)]
        self.train_lr = train_lr

        self.pretrain_modules = [MultiHeadModule(encoder=nn.Sequential(enc.shared,
                                                                       spec),
                                                 heads_list=[nn.Linear(embedding_dim, 1)
                                                            for _ in range(len(pretrain_metrics))])
                                 for enc in self.encoders for spec in enc.specialized_list]
        self.pretrain_optimizers = [torch.optim.Adam(module.parameters(), lr=self.pretrain_lr)
                                    for module in self.pretrain_modules]

        self.modules = [MultiHeadModule(encoder=enc.shared,
                                        heads_list=[nn.Sequential(spec,
                                                                 nn.Linear(embedding_dim, 1))
                                                   for spec in enc.specialized_list])
                        for enc in self.encoders]
        self.optimizers = [torch.optim.Adam(module.parameters(), lr=self.train_lr) for module in self.modules]

    def train_multiple(self, input_data, target_data, epochs, bs=16):
        train_set = TensorDataset(input_data, *target_data)
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)

        def train_epoch(module, optimizer):
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                preds = module(batch[0])
                loss = 0
                for i in range(1, len(batch)):
                    loss += nn.functional.huber_loss(input=preds[i - 1], target=batch[i].view(preds[i - 1].shape))
                loss.backward()
                optimizer.step()
            return module, optimizer

        with Parallel(n_jobs=self.devices) as parallel:
            for _ in range(epochs):
                res = parallel(
                    delayed(train_epoch)(
                        module,
                        optimizer
                    )
                    for module, optimizer in zip(self.modules, self.optimizers)
                )
                self.modules, self.optimizers = [], []
                for (module, optimizer) in res:
                    self.modules.append(module)
                    self.optimizers.append(optimizer)

    def pretrain(self):
        train_set = TensorDataset(self.pretrain_configs, *self.pretrain_metrics)
        train_loader = DataLoader(train_set, batch_size=self.pretrain_bs, shuffle=True, num_workers=0)

        def pretrain_epoch(module, optimizer):
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                preds = module(batch[0])
                loss = 0
                for i in range(1, len(batch)):
                    loss += nn.functional.huber_loss(input=preds[i - 1], target=batch[i].view(preds[i - 1].shape))
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
        split_list = [self.pretrain_modules[i:i+self.n_objectives] for i in range(self.n_networks)]
        self.modules = []
        for lst in split_list:
            encoder = lst[0].get_submodule('encoder').get_submodule('0')
            heads_list = [nn.Sequential(e.get_submodule('encoder').get_submodule('1'),
                                        nn.Linear(self.embedding_dim, 1))
                          for e in lst]
            self.modules.append(MultiHeadModule(encoder=encoder,
                                                heads_list=heads_list))

        self.optimizers = [torch.optim.Adam(module.parameters(), lr=self.train_lr) for module in self.modules]

    def predict(self, input_data):
        result = []
        for idx in range(self.n_objectives):
            with torch.no_grad():
                preds = torch.stack([module(input_data)[idx] for module in self.modules])
                result.append(preds.mean(dim=0).detach().cpu().numpy())
        return result