import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.utils as utils


class LibriSamples(torch.utils.data.Dataset):
    def __init__(self, data_path, cls_path):
        self.X_dir = data_path
        self.Y_dir = cls_path
        self.X = np.load(data_path)
        self.Y = np.load(cls_path)

    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, item):
        x = self.X[item]
        y = self.Y[item]
        return x, y


train_data_path = "data/train_bags_demo.npy"
train_cls_path = "data/train_clsname_demo.npy"
valid_data_path = "data/valid_bags_demo.npy"
valid_cls_path = "data/valid_clsname_demo.npy"

batch_size = 32
# print(len(np.load(train_cls_path)))
# print(len(np.load(train_data_path)))
# print(len(np.load(valid_cls_path)))
# print(len(np.load(valid_data_path)))

train_data = LibriSamples(train_data_path, train_cls_path)
valid_data = LibriSamples(valid_data_path, valid_cls_path)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

print(len(train_loader))
print(len(valid_loader))

for data in train_loader:
    x, y = data
    print(x)
    print(y)
    print(x.shape, y.shape)
    break
# for data in valid_loader:
#     x, y = data
#     print(x)
#     print(y)
#     print(x.shape, y.shape)
#     break
