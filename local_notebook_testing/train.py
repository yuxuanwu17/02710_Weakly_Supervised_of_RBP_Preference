import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from notebook.model import WeakRM


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


train_data_path = "../data/train_bags_demo.npy"
train_cls_path = "../data/train_clsname_demo.npy"
valid_data_path = "../data/valid_bags_demo.npy"
valid_cls_path = "../data/valid_clsname_demo.npy"

batch_size = 1
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

# for data in train_loader:
#     x, y = data
#     print(x)
#     print(y)
#     print(x.shape, y.shape)
#     break

model = WeakRM()
epochs = 3
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, epochs + 1):
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    num_correct = 0
    total_loss = 0

    # training samples
    # model.train()
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x
        y = y

        # Don't be surprised - we just wrap these two lines to make it work for FP16
        outputs = model(x)
        loss = criterion(outputs, y)

        num_correct += int((torch.argmax(outputs) == y).sum())
        total_loss += float(loss)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        # Another couple things you need for FP16.
        loss.backward()  # This is a replacement for loss.backward()
        optimizer.step()  # This is a replacement for optimizer.step()

        batch_bar.update()  # Update tqdm bar
    batch_bar.close()  # You need this to close the tqdm bar

