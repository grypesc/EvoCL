import copy
import random
import torch
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet32 import resnet8, resnet14, resnet20, resnet32
from .incremental_learning import Inc_Learning_Appr
from .criterions.proxy_nca import ProxyNCA
from .criterions.ce import CE

torch.backends.cuda.matmul.allow_tf32 = False

class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, S=64, adapter="linear", criterion="proxy-nca", alpha=0.5, smoothing=0., sval_fraction=0.95, adapt=False, activation_function="relu", nnet="resnet32"):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.S = S
        self.alpha = alpha
        self.smoothing = smoothing

        self.activation = activation_function
        self.model_class = {"resnet8": resnet8,
                            "resnet14": resnet14,
                            "resnet20": resnet20,
                            "resnet32": resnet32}[nnet]
        self.models = nn.ModuleList()
        self.model = None
        self.train_data_loaders, self.val_data_loaders = [], []
        self.prototypes = {}
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.criterion = {"proxy-nca": ProxyNCA,
                          "ce" : CE}[criterion]
        self.adapt = adapt
        self.sval_fraction = sval_fraction
        self.svals_explained_by = []
        self.adapter_type = adapter


    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--S',
                            help='latent space size',
                            type=int,
                            default=64)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=0.5)
        parser.add_argument('--sval-fraction',
                            help='Fraction of eigenvalues sum that is explained',
                            type=float,
                            default=0.95)
        parser.add_argument('--adapt',
                            help='Adapt prototypes',
                            action='store_true',
                            default=False)
        parser.add_argument('--activation-function',
                            help='Activation functions in resnet',
                            type=str,
                            choices=["identity", "relu", "lrelu"],
                            default="relu")
        parser.add_argument('--adapter',
                            help='adapter',
                            type=str,
                            choices=["linear", "mlp"],
                            default="linear")
        parser.add_argument('--criterion',
                            help='Loss function',
                            type=str,
                            choices=["ce", "proxy-nca"],
                            default="proxy-nca")
        parser.add_argument('--smoothing',
                            help='label smoothing',
                            type=float,
                            default=0.0)
        parser.add_argument('--nnet',
                            type=str,
                            choices=["resnet8", "resnet14", "resnet20", "resnet32"],
                            default="resnet32")
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])

        print("### Training backbone ###")
        self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)
        if t > 0 and self.adapt:
            print("### Adapting prototypes ###")
            self.adapt_prototypes(t, trn_loader, val_loader)
        print("### Creating new prototypes ###")
        self.create_prototypes(t, trn_loader, val_loader, num_classes_in_t)
        self.check_singular_values(t, val_loader)
        self.print_singular_values()


    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        for model in self.models:
            model.eval()
        model = self.model_class(num_features=self.S, activation_function=self.activation)
        self.models.append(model)
        model = model.to(self.device, non_blocking=True)
        print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} frozen parameters\n')

        adapter = nn.Linear((t+1) * self.S, t * self.S)
        if self.adapter_type == "mlp":
            adapter = nn.Sequential(nn.Linear((t+1) * self.S, 2 * t * self.S),
                                    nn.GELU(),
                                    nn.Linear(2 * t * self.S, t * self.S)
                                    )
        adapter.to(self.device, non_blocking=True)

        criterion = self.criterion(num_classes_in_t, self.S * (t+1), self.device)
        parameters = list(model.parameters()) + list(criterion.parameters()) + list(adapter.parameters())
        optimizer, lr_scheduler = self.get_optimizer(parameters, self.wd)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
            train_hits, val_hits = 0, 0
            model.train()
            adapter.train()
            criterion.train()

            for images, targets in trn_loader:
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                old_features = None
                features = model(images)
                if t > 0:
                    with torch.no_grad():
                        old_features = [model(images) for model in self.models[:-1]]
                        old_features = torch.cat(old_features, dim=1)
                    features = torch.cat((old_features, features), dim=1)

                loss, logits = criterion(features, targets)
                total_loss, kd_loss = self.distill_knowledge(loss, features, adapter, old_features)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.clipgrad)

                optimizer.step()
                if logits is not None:
                    train_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
                train_kd_loss.append(float(kd_loss))
            lr_scheduler.step()

            model.eval()
            adapter.eval()
            criterion.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    old_features = None
                    features = model(images)
                    if t > 0:
                        old_features = [model(images) for model in self.models[:-1]]
                        old_features = torch.cat(old_features, dim=1)
                        features = torch.cat((old_features, features), dim=1)
                    loss, logits = criterion(features, targets)
                    _, kd_loss = self.distill_knowledge(loss, features, adapter, old_features)
                    if logits is not None:
                        val_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                        valid_kd_loss.append(float(kd_loss))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            train_kd_loss = sum(train_kd_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            valid_kd_loss = sum(valid_kd_loss) / len(val_loader.dataset)

            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} "
                  f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")


    @torch.no_grad()
    def create_prototypes(self, t, trn_loader, val_loader, num_classes_in_t):
        """ Create distributions for task t"""
        for model in self.models:
            model.eval()
        transforms = val_loader.dataset.transform
        for c in range(num_classes_in_t):
            c = c + self.task_offset[t]
            train_indices = torch.tensor(trn_loader.dataset.labels) == c
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), (t+1) * self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = [model(images) for model in self.models]
                features = torch.cat(features, dim=1)
                class_features[from_: from_+bsz] = features
                flipped_images = torch.flip(images, dims=(3,))
                features = [model(flipped_images) for model in self.models]
                features = torch.cat(features, dim=1)
                class_features[from_+bsz: from_+2*bsz] = features
                from_ += 2*bsz

            # Calculate centroid
            centroid = class_features.mean(dim=0)
            self.prototypes[c] = centroid

        print("Proto norm statistics:")
        protos = torch.norm(torch.stack(list(self.prototypes.values())), dim=1)
        print(f"Mean: {protos.mean():.2f}, median: {protos.median():.2f}")
        print(f"Range: [{protos.min():.2f}; {protos.max():.2f}]")

    def adapt_prototypes(self, t, trn_loader, val_loader):
        # First, train the adapter
        for model in self.models:
            model.eval()
        adapter = nn.Linear(t * self.S, self.S)
        if self.adapter_type == "mlp":
            adapter = nn.Sequential(nn.Linear(t * self.S, 2 * t * self.S),
                                    nn.GELU(),
                                    nn.Linear(2 * t * self.S, self.S)
                                    )
        adapter.to(self.device, non_blocking=True)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        for epoch in range(self.nepochs):
            adapter.train()
            train_loss, valid_loss = [], []
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.no_grad():
                    features = [model(images) for model in self.models[:-1]]
                    features = torch.cat(features, dim=1)
                    target = self.models[-1](images)
                adapted_features = adapter(features)
                loss = torch.nn.functional.mse_loss(adapted_features, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), self.clipgrad)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            adapter.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    bsz = images.shape[0]
                    images = images.to(self.device, non_blocking=True)
                    features = [model(images) for model in self.models[:-1]]
                    features = torch.cat(features, dim=1)
                    target = self.models[-1](images)
                    adapted_features = adapter(features)
                    loss = torch.nn.functional.mse_loss(adapted_features, target)
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {100*train_loss:.2f} Val loss: {100*valid_loss:.2f} ")

        # Calculate new dimension values for old prototypes
        with torch.no_grad():
            adapter.eval()
            for c, prototype in self.prototypes.items():
                new_vals = adapter(prototype)
                self.prototypes[c] = torch.cat((prototype, new_vals), dim=0)


    @torch.no_grad()
    def eval(self, t, val_loader):
        """ Perform nearest centroids classification """
        for model in self.models:
            model.eval()
        prototypes = torch.stack(list(self.prototypes.values()))
        tag_acc = Accuracy("multiclass", num_classes=prototypes.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = [model(images) for model in self.models]
            features = torch.cat(features, dim=1)
            dist = torch.cdist(features, prototypes)
            tag_preds = torch.argmin(dist, dim=1)
            taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset
            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    def distill_knowledge(self, loss, features, distiller, old_features=None):
        """Returns loss ce with kd"""
        if old_features is None:
            return loss, 0
        kd_loss = nn.functional.mse_loss(distiller(features), old_features)
        total_loss = (1 - self.alpha) * loss + self.alpha * kd_loss
        return total_loss, kd_loss

    def get_optimizer(self, parameters, wd):
        """Returns the optimizer"""
        milestones = [int(self.nepochs * 0.3), int(self.nepochs * 0.6), int(self.nepochs * 0.9)]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler


    def get_adapter_optimizer(self, parameters):
        """Returns the optimizer"""
        milestones = [int(self.nepochs * 0.3), int(self.nepochs * 0.6), int(self.nepochs * 0.9)]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    @torch.no_grad()
    def check_singular_values(self, t, val_loader):
        for model in self.models:
            model.eval()
        self.svals_explained_by.append([])
        for i, _ in enumerate(self.train_data_loaders):
            ds = ClassMemoryDataset(self.train_data_loaders[i].dataset.images, val_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=val_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((len(ds), (t+1) * self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = [model(images) for model in self.models]
                features = torch.cat(features, dim=1)
                class_features[from_: from_ + bsz] = features
                from_ += bsz

            cov = torch.cov(class_features.T)
            svals = torch.linalg.svdvals(cov)
            xd = torch.cumsum(svals, 0)
            xd = xd[xd < self.sval_fraction * torch.sum(svals)]
            explain = xd.shape[0]
            self.svals_explained_by[t].append(explain)

    def print_singular_values(self):
        print(f"{self.sval_fraction} of eigenvalues sum is explained by:")
        for t, explained_by in enumerate(self.svals_explained_by):
            print(f"Task {t}: {explained_by}")
