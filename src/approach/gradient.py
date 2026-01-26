import copy
import torch
import numpy as np
import torchmetrics

from argparse import ArgumentParser
from itertools import compress, cycle
from torch import nn
from torch.utils.data import DataLoader

from .evo_utils.feature_dataset import FeatureDataset
from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .incremental_learning import Inc_Learning_Appr
from .models.resnet18_for_evo import resnet18


class NeuralNet(nn.Module):
    def __init__(self, model_type, S, adapter_type, device):
        super().__init__()
        self.S = S
        self.model_type = model_type
        self.backbone = nn.Sequential(
            nn.Conv2d(3, S, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(S, S, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(S, S, 3, padding=1),
            nn.GELU()
            )
        self.head = None
        self.adapter_type = adapter_type
        self.adapter = None
        self.device = device

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = nn.functional.normalize(x)
        return x

    def calculate_logits(self, x):
        return self.head(x)

    def reset(self, output_size):
        device = self.device
        self.head = nn.Linear(self.S, output_size, device=device)
        self.adapter = nn.Linear(self.S, self.S, device=device)
        nn.init.eye_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)  # Set the bias to zero
        if self.adapter_type == "mlp":
            self.adapter = nn.Sequential(
                nn.Linear(self.S, 16, device=device),
                nn.GELU(),
                nn.Linear(16, self.S, device=device),
            )

    def adapt(self, x):
        x = x + self.adapter(x)
        return nn.functional.normalize(x)

    def load_weights_from_path(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)


class Appr(Inc_Learning_Appr):
    def __init__(self, model, device, **kwargs):
        super(Appr, self).__init__(model, device, **kwargs)

        self.S = kwargs["S"]
        self.margin = kwargs["margin"]
        self.dump = kwargs["dump"]
        self.alpha = kwargs["alpha"]
        self.loss_type = kwargs["loss_type"]
        self.is_rotation = kwargs["rotation"]

        self.adapter = kwargs["adapter"]
        self.old_model = None
        self.model = NeuralNet(kwargs["nnet"], self.S, self.adapter, self.device)
        self.model.to(device, non_blocking=True)
        self.is_pretrained = False
        if kwargs["load_model_path"] is not None:
            self.model.load_weights_from_path(kwargs["load_model_path"])
            self.is_pretrained = True
        self.model.eval()

        self.train_data_loaders, self.val_data_loaders = [], []
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.feature_dataset = FeatureDataset(self.S, device)
        self.features_per_class = 64

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--S',
                            help='latent space size',
                            type=int,
                            default=32)
        parser.add_argument('--margin',
                            help='Margin for contrastive loss',
                            type=float,
                            default=2)
        parser.add_argument('--alpha',
                            help='Weight of adapter loss',
                            type=float,
                            default=10)
        parser.add_argument('--nnet',
                            help='Type of neural network',
                            type=str,
                            choices=["mlp", "resnet18", "convnet"],
                            default="resnet18")
        parser.add_argument('--loss-type',
                            type=str,
                            choices=["ce", "contrastive"],
                            default="ce")
        parser.add_argument('--adapter',
                            help='Type of neural network',
                            type=str,
                            choices=["linear", "mlp"],
                            default="linear")
        parser.add_argument('--dump',
                            help='save checkpoints',
                            action='store_true',
                            default=False)
        parser.add_argument('--load-model-path',
                            help='Path to model trained with gradient method, to initialize weights"',
                            type=str,
                            default=None)
        parser.add_argument('--rotation',
                            help='Rotate images in the first task to enhance feature extractor',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.model.reset(sum(self.classes_in_tasks))
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        print(f"### Training model task: {t} ###")
        if t > 0 or not self.is_pretrained:
            print(f"### Training model task: {t} ###")
            self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)
        if t > 0:
            print(f"### Adapting features task: {t} ###")
            self.adapt_features(0)
        if self.dump:
            torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        print(f"### Creating new features  task: {t} ###\n")
        self.store_features(t, trn_loader, val_loader, num_classes_in_t)

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        trn_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        if t > 0:
            feature_loader = DataLoader(self.feature_dataset, batch_size=trn_loader.batch_size, num_workers=0, shuffle=True, drop_last=True)
            feature_iter = cycle(feature_loader)
        if t == 0 and self.is_rotation:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size // 4, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
            self.model.head = nn.Linear(self.S, 4 * num_classes_in_t, device=self.device)
        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} parameters')
        past_features, past_targets, old_features = None, None, None
        parameters = list(self.model.parameters())
        optimizer, lr_scheduler = self.get_optimizer(parameters, t, self.wd)
        for epoch in range(self.nepochs):
            loss_metric = torchmetrics.MeanMetric()
            ce_loss_metric = torchmetrics.MeanMetric()
            approx_loss_metric = torchmetrics.MeanMetric()
            adapter_loss_metric = torchmetrics.MeanMetric()
            self.model.train()
            for images, targets in trn_loader:
                optimizer.zero_grad()
                images, targets = images.to(self.device), targets.to(self.device)
                if t == 0 and self.is_rotation:
                    images, targets = compute_rotations(images, targets, num_classes_in_t)
                if t > 0:
                    past_features, past_targets = next(feature_iter)
                    old_features = self.old_model(images)

                total_loss, ce_loss, approx_loss, adapter_loss = self.calculate_loss(images, targets, past_features, past_targets, old_features)
                loss_metric.update(float(total_loss))
                ce_loss_metric.update(float(ce_loss))
                approx_loss_metric.update(float(approx_loss))
                adapter_loss_metric.update(1000 * float(adapter_loss))
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
            lr_scheduler.step()
            #  Evaluation
            self.model.eval()
            print(f"Epoch: {epoch}: Loss: {loss_metric.compute():.2f} CE: {ce_loss_metric.compute():.2f} ApproxCE: {approx_loss_metric.compute():.2f} Adapter: {adapter_loss_metric.compute():.3f} ", end="")
            if t == 1:
                self.eval_old_new_losses()
            else:
                print("")


    def calculate_loss(self, images, targets, past_features, past_targets, old_features):
        if past_features is None:
            features = self.model(images)
            if self.loss_type == "ce":
                logits = self.model.calculate_logits(features)
                loss = nn.functional.cross_entropy(logits, targets)
                return loss, loss, 0, 0
            # dist = torch.cdist(features, features)
            # similarity_mask = targets.unsqueeze(1) == targets.unsqueeze(0)
            # loss_pull = dist[similarity_mask].pow(2).mean()
            # loss_push = torch.clamp(self.margin - dist[~similarity_mask], min=0).pow(2).mean()
            # loss = loss_push + loss_pull
            # return loss, 0, -1, 0, loss_push, loss_pull

        features = self.model(images)
        adapted_old_features = self.model.adapt(old_features)
        adapter_loss = self.S * nn.functional.mse_loss(adapted_old_features, features)

        targets = torch.cat((targets, past_targets))
        adapted_features = self.model.adapt(past_features)
        all_features = torch.cat((features, adapted_features), dim=0)

        if self.loss_type == "ce":
            logits = self.model.calculate_logits(all_features)
            loss = nn.functional.cross_entropy(logits, targets)
            approx_loss = nn.functional.cross_entropy(logits[features.shape[0]:], targets[features.shape[0]:])
            return loss + self.alpha * adapter_loss, loss, approx_loss, adapter_loss

        # dist = torch.cdist(all_features, all_features)
        # similarity_mask = targets.unsqueeze(1) == targets.unsqueeze(0)
        # loss_pull = dist[similarity_mask].pow(2).mean()
        # loss_push = torch.clamp(self.margin - dist[~similarity_mask], min=0).pow(2).mean()
        # loss = loss_push + loss_pull + self.alpha * adapter_loss
        # return loss, 0, -1, adapter_loss, loss_push, loss_pull

    def get_optimizer(self, parameters, t, wd):
        """Returns the optimizer"""
        milestones = (int(0.3*self.nepochs), int(0.6*self.nepochs), int(0.9*self.nepochs))
        lr = self.lr
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    @torch.no_grad()
    def adapt_features(self, num_workers):
        print(f"Features norm before adaptation: {self.feature_dataset.features.norm(dim=1).mean():.3f}")
        loader = DataLoader(self.feature_dataset, batch_size=512, num_workers=num_workers, shuffle=False)
        new_features = []

        for features, labels in loader:
            new_features.append(self.model.adapt(features))
        new_features = torch.cat(new_features, dim=0)
        self.feature_dataset.features = new_features
        print(f"Features norm after adaptation: {self.feature_dataset.features.norm(dim=1).mean():.3f}")

    @torch.no_grad()
    def store_features(self, t, trn_loader, val_loader, num_classes_in_t):
        transforms = val_loader.dataset.transform

        for c in range(num_classes_in_t):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[t]
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                train_images = train_images[:self.features_per_class]
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ds[:self.features_per_class]
                ds = ClassMemoryDataset(ds, transforms)
            loader = DataLoader(ds, batch_size=512, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_+bsz] = features
                from_ += bsz
            self.feature_dataset.add(class_features[:self.features_per_class], torch.full((self.features_per_class, ), fill_value=c + self.task_offset[t], device=self.device))

        print(f"New features norm: {self.feature_dataset.features[-num_classes_in_t*self.features_per_class:].norm(dim=1).mean():.3f}")

    @torch.no_grad()
    def eval(self, t, val_loader):
        self.model.eval()
        tag_acc = torchmetrics.Accuracy("multiclass", num_classes=sum(self.classes_in_tasks))
        taw_acc = torchmetrics.Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            logits = self.model.calculate_logits(features)
            tag_preds = logits.argmax(1)
            taw_preds = logits[:, self.task_offset[t]: self.task_offset[t+1]].argmax(1)
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
            old_loss = nn.functional.cross_entropy(old_logits, old_targets)

            # new_images, new_targets = next(new_iter)
            # new_images, new_targets = new_images.to(self.device), new_targets.to(self.device)
            # new_features = self.model(new_images)
            # new_logits = self.model.calculate_logits(new_features)
            # new_loss = nn.functional.cross_entropy(new_logits[0], new_targets)

            old_loss_metric.update(float(old_loss))
            # new_loss_metric.update(float(new_loss))
        print(f"Old: {old_loss_metric.compute():.2f}")


def compute_rotations(images, targets, total_classes):
    # compute self-rotation for the first task following PASS https://github.com/Impression2805/CVPR21_PASS
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets
