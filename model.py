import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
from torchsummaryX import summary


class WeakRM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(40, 32, kernel_size=15, padding=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(32, 16, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Flatten()
        )

        self.attention_v = nn.Sequential(
            nn.Linear(32, 128),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(32, 128),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(
            nn.Linear(128, 1),
            nn.Softmax()
        )

        self.cls = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, training=True, mask=None):
        inputs = torch.squeeze(inputs, 0)

        embedding = self.embedding(inputs)  # torch.Size([13, 32])
        print(embedding.shape)

        attention_v = self.attention_v(embedding)
        attention_u = self.attention_v(embedding)

        print(attention_u.shape, attention_v.shape)
        gated_attention = self.attention_weights(attention_u * attention_v).permute((1, 0))

        gated_attention = nn.Softmax()(gated_attention)  # torch.Size([1, 13])

        bag_features = torch.matmul(gated_attention, embedding)
        print(embedding.shape, gated_attention.shape)
        bag_probability = self.cls(bag_features)

        return bag_probability, gated_attention


if __name__ == '__main__':
    x = torch.rand((1, 13, 40, 5))
    encoder = WeakRM()
    summary(encoder, x)
