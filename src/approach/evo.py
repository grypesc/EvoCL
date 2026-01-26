import copy
import random
import torch
import numpy as np
import torchmetrics

from argparse import ArgumentParser
from itertools import compress, cycle

from torch import nn
from torch.utils.data import DataLoader

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .incremental_learning import Inc_Learning_Appr
from .evo_utils.feature_dataset import FeatureDataset
from .evo_utils.lambda_strategy import LambdaStrategy
from .evo_utils.vit_strategy import ViTStrategy
from .evo_utils.mu_plus_lambda_strategy import MuPlusLambdaStrategy
from .evo_utils.differential_strategy import DifferentialStrategy
from .evo_utils.resnet18_strategy import Resnet18Strategy


def maybe_cast_to_fp16(func):
    def wrapper(*args):
        if not args[0].use_fp16:
            return func(*args)
        with torch.autocast(device_type="cuda"):
            return func(*args)
    return wrapper


class EvoOptimizer:
    def __init__(self, population, t, num_classes, max_epochs, mu_, lambda_, lr, lr_ratio, wd, device):
        population.prepare_for_training(mu_, num_classes, t)
        self.population = population
        self.t = t
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.wd = wd
        self.lr = lr
        self.start_lr = lr
        self.end_lr = lr_ratio * self.lr
        self.lr_step = (self.lr - self.end_lr) / (max_epochs - 1)
        self.device = device
        self.epoch = 0
        self.max_epochs = max_epochs
        self.total_loss_metric = torchmetrics.MeanMetric()
        self.ce_loss_metric = torchmetrics.MeanMetric()
        self.approx_ce_loss_metric = torchmetrics.MeanMetric()
        self.adapter_loss_metric = torchmetrics.MeanMetric()
        self.wd_loss_metric = torchmetrics.MeanMetric()

    def step(self, images, targets, past_features, past_targets, old_features):
        self.population.create_offspring(self.t, self.lambda_, self.lr, mutate_only_head=self.epoch < 20)
        features = self.population(images)
        total_loss, ce_loss, approx_ce_loss, adapter_loss, wd_loss = self.population.calculate_loss(features, targets, past_features, past_targets, old_features, self.wd)
        self.total_loss_metric.update(float(total_loss[0]))
        self.ce_loss_metric.update(float(ce_loss[0]))
        self.approx_ce_loss_metric.update(float(approx_ce_loss[0]))
        self.adapter_loss_metric.update(1000 * float(adapter_loss[0]))
        self.wd_loss_metric.update(float(wd_loss[0]))
        self.population.survive(self.mu_, total_loss)

    def on_epoch_end(self):
        self.lr -= self.lr_step
        print(f"Epoch: {self.epoch}, Total: {self.total_loss_metric.compute():.2f} CE: {self.ce_loss_metric.compute():.2f}  ApproxCE: {self.approx_ce_loss_metric.compute():.2f} "
              f"Adapter: {self.adapter_loss_metric.compute():.3f} Wd: {self.wd_loss_metric.compute():.2f} ", end="")
        self.epoch += 1
        # if self.epoch == int(self.max_epochs * 0.5):
        #     self.lr = self.start_lr
        self.total_loss_metric.reset()
        self.ce_loss_metric.reset()
        self.approx_ce_loss_metric.reset()
        self.adapter_loss_metric.reset()
        self.wd_loss_metric.reset()


class Appr(Inc_Learning_Appr):
    def __init__(self, model, device, **kwargs):
        super(Appr, self).__init__(model, device, **kwargs)
        self.S = kwargs["S"]
        self.alpha = kwargs["alpha"]
        self.mu_ = kwargs["mu"]
        self.lambda_ = kwargs["lamb"]
        self.lr_ratio = kwargs["lr_ratio"]
        self.use_fp16 = kwargs["use_fp16"]
        self.max_epochs = self.nepochs
        self.old_model = None

        class_ = {"Differential": DifferentialStrategy,
                  "MuPlusLambda": MuPlusLambdaStrategy,
                  "Lambda": LambdaStrategy,
                  "vit": ViTStrategy,
                  "Resnet18": Resnet18Strategy}[kwargs["strategy"]]
        self.train_on_0th_task = False
        if kwargs["strategy"] == "vit":
            self.train_on_0th_task = True
            self.S = 384
        self.model = class_(self.S, kwargs["F"], self.alpha, kwargs["hidden_size"], device)
        if kwargs["load_model_path"] is not None:
            self.model.load_weights_from_path(kwargs["load_model_path"])

        self.train_data_loaders, self.val_data_loaders = [], []
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.feature_dataset = FeatureDataset(self.S, device)
        self.features_per_class = kwargs["K"]

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--S',
                            help='latent space size',
                            type=int,
                            default=32)
        parser.add_argument('--K',
                            help='k for knn classifier',
                            type=int,
                            default=128)
        parser.add_argument('--F',
                            help='latent space size',
                            type=float,
                            default=1.0)
        parser.add_argument('--alpha',
                            help='Weight of adapter loss',
                            type=float,
                            default=100)
        parser.add_argument('--lr-ratio',
                            help='Weight of adapter loss',
                            type=float,
                            default=0.1)
        parser.add_argument('--mu',
                            help='Weight of adapter loss',
                            type=int,
                            default=16)
        parser.add_argument('--hidden-size',
                            help='xx',
                            type=int,
                            default=16)
        parser.add_argument('--lamb',
                            help='Weight of adapter loss',
                            type=int,
                            default=128)
        parser.add_argument('--strategy',
                            help='Population type',
                            type=str,
                            choices=["Lambda", "MuPlusLambda", "Differential", "MLP", "Resnet18", "vit"],
                            default="Lambda")
        parser.add_argument('--load-model-path',
                            help='Path to model trained with gradient method, to initialize weights"',
                            type=str,
                            default=None)
        parser.add_argument('--use-fp16',
                            help='Use FP16 precision',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    @maybe_cast_to_fp16
    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])
        self.old_model = copy.deepcopy(self.model)
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        if t > 0 or self.train_on_0th_task:
            print(f"### Training model task: {t} ###")
            self.train_backbone(t, trn_loader)
        if t > 0:
            print(f"\n### Adapting features task: {t} ###")
            self.adapt_features(0)
        print(f"\n### Creating new features  task: {t} ###")
        self.store_features(trn_loader, val_loader, num_classes_in_t)

    def train_backbone(self, t, trn_loader):
        trn_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        if t > 0:
            feature_loader = DataLoader(self.feature_dataset, batch_size=t * trn_loader.batch_size, num_workers=0, shuffle=True, drop_last=True)
            feature_iter = cycle(feature_loader)
        optimizer = EvoOptimizer(self.model, t, sum(self.classes_in_tasks), self.max_epochs, self.mu_, self.lambda_, self.lr, self.lr_ratio, self.wd, self.device)
        past_features, past_targets, old_features = None, None, None
        for epoch in range(self.nepochs):
            for images, targets in trn_loader:
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                if t > 0:
                    past_features, past_targets = next(feature_iter)
                    old_features = self.old_model(images)
                optimizer.step(images, targets, past_features, past_targets, old_features)
            optimizer.on_epoch_end()
            #  Evaluation
            if t == 1:
                self.eval_old_new_losses()
            else:
                print("")
        self.model.finish_training()

    def adapt_features(self, num_workers):
        print(f"Features norm before adaptation: {self.feature_dataset.features.norm(dim=1).mean():.3f}")
        loader = DataLoader(self.feature_dataset, batch_size=512, num_workers=num_workers, shuffle=False)
        new_features = []

        for features, labels in loader:
            new_features.append(self.model.adapt(features.unsqueeze(0)).squeeze(0))
        new_features = torch.cat(new_features, dim=0)
        self.feature_dataset.features = new_features
        print(f"Features norm after adaptation: {self.feature_dataset.features.norm(dim=1).mean():.3f}")

    def store_features(self, trn_loader, val_loader, num_classes_in_t):
        transforms = val_loader.dataset.transform
        for c in set(trn_loader.dataset.labels):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = DataLoader(ds, batch_size=512, num_workers=trn_loader.num_workers, shuffle=True)
            from_ = 0
            class_features = torch.zeros((len(ds), self.S), device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images).squeeze(0)
                class_features[from_: from_+bsz] = features
                from_ += bsz
            self.feature_dataset.add(class_features[:self.features_per_class], torch.full((from_, ), fill_value=c, device=self.device)[:self.features_per_class])
        print(f"New features norm: {self.feature_dataset.features[-num_classes_in_t*self.features_per_class:].norm(dim=1).mean():.3f}")

    def eval(self, t, loader):
        tag_acc = torchmetrics.Accuracy("multiclass", num_classes=sum(self.classes_in_tasks))
        taw_acc = torchmetrics.Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            logits = self.model.calculate_logits(features).squeeze(0)
            tag_preds = logits.argmax(1)
            taw_preds = logits[:, self.task_offset[t]: self.task_offset[t+1]].argmax(1)

            # dist = - features.squeeze(1) @ self.feature_dataset.features.T
            # # dist = torch.cdist(features, self.feature_dataset.features.unsqueeze(0)).squeeze(1)
            # _, lowest = torch.topk(dist, self.K, 1, largest=False)
            # tag_preds, _ = self.feature_dataset.labels[lowest].mode(1)
            # _, lowest = torch.topk(dist[:, self.features_per_class * offset: self.features_per_class * (offset + self.classes_in_tasks[t])], self.K, 1, largest=False)
            # taw_preds, _ = self.feature_dataset.labels[lowest].mode(1)
            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu() + offset, target)
        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    def eval_old_new_losses(self):
        old_loader = self.val_data_loaders[-2]
        # new_loader = self.val_data_loaders[-1]
        # new_iter = cycle(new_loader)

        old_loss_metric = torchmetrics.MeanMetric()
        # new_loss_metric = torchmetrics.MeanMetric()

        for old_images, old_targets in old_loader:
            old_images, old_targets = old_images.to(self.device), old_targets.to(self.device)

            old_features = self.model(old_images)
            old_logits = self.model.calculate_logits(old_features)
            old_loss = nn.functional.cross_entropy(old_logits[0], old_targets)

            # new_images, new_targets = next(new_iter)
            # new_images, new_targets = new_images.to(self.device), new_targets.to(self.device)
            # new_features = self.model(new_images)
            # new_logits = self.model.calculate_logits(new_features)
            # new_loss = nn.functional.cross_entropy(new_logits[0], new_targets)

            old_loss_metric.update(float(old_loss))
            # new_loss_metric.update(float(new_loss))
        print(f"Old: {old_loss_metric.compute():.2f}")