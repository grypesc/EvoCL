import torch


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feature_size, device):
        self.features = torch.zeros((0, feature_size), device=device)
        self.labels = torch.zeros((0,), dtype=torch.int64, device=device)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def add(self, new_features, new_labels):
        self.labels = torch.cat((self.labels, new_labels), dim=0)
        self.features = torch.cat((self.features, new_features), dim=0)