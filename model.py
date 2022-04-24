import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
from torchsummaryX import summary


# class WeakRM(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.embedding = nn.Sequential(
#             nn.Conv1d(40, 32, kernel_size=15, padding=7, stride=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(0.25),
#             nn.Conv1d(32, 16, kernel_size=5, padding=2, stride=1),
#             nn.ReLU(),
#             nn.Dropout(0.25),
#             nn.Flatten()
#         )
#
#         self.attention_v = nn.Sequential(
#             nn.Linear(32, 128),
#             nn.Tanh()
#         )
#
#         self.attention_u = nn.Sequential(
#             nn.Linear(32, 128),
#             nn.Sigmoid()
#         )
#
#         self.attention_weights = nn.Sequential(
#             nn.Linear(128, 1),
#             nn.Softmax()
#         )
#
#         self.cls = nn.Sequential(
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, inputs, training=True, mask=None):
#         inputs = torch.squeeze(inputs, 0)
#
#         embedding = self.embedding(inputs)  # torch.Size([13, 32])
#         # print(embedding.shape)
#
#         attention_v = self.attention_v(embedding)
#         attention_u = self.attention_v(embedding)
#
#         # print(attention_u.shape, attention_v.shape)
#         gated_attention = self.attention_weights(attention_u * attention_v).permute((1, 0))
#
#         gated_attention = nn.Softmax()(gated_attention)  # torch.Size([1, 13])
#
#         bag_features = torch.matmul(gated_attention, embedding)
#         # print(embedding.shape, gated_attention.shape)
#         bag_probability = self.cls(bag_features)
#
#         return bag_probability, gated_attention
class WeakRM(nn.Module):
    def __init__(self, training=True):
        super().__init__()

        self.inst_conv1 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=15, padding=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.inst_conv2 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2, stride=1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.2)

        self.attention_v = nn.Sequential(
            nn.Linear(320, 128),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(320, 128),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(
            nn.Linear(128, 1),
            nn.Softmax()
        )

        self.cls = nn.Sequential(
            nn.Linear(320, 1),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax()

    def forward(self, inputs, training=True, mask=None):
        inputs = torch.squeeze(inputs, 0)
        print(inputs.shape)
        inputs = inputs.permute((0, 2, 1))
        print(inputs.shape)
        inst_conv1 = self.inst_conv1(inputs)  # torch.Size([13, 32, 20])
        print(inst_conv1.shape)
        if training:
            inst_conv1 = self.dropout(inst_conv1)

        inst_conv2 = self.inst_conv2(inst_conv1)
        print(inst_conv2.shape)

        inst_features = nn.Flatten()(inst_conv2)
        print(inst_features.shape)  # torch.Size([13, 320])

        attention_v = self.attention_v(inst_features)
        attention_u = self.attention_v(inst_features)

        print(attention_u.shape)
        print(attention_v.shape)

        # print(attention_u.shape, attention_v.shape)
        gated_attention = self.attention_weights(attention_u * attention_v).permute((1, 0))
        print(gated_attention.shape)

        gated_attention = self.softmax(gated_attention)  # torch.Size([1, 13])
        print("gated attention shape", gated_attention.shape)
        print("inst features shape", inst_features.shape)

        bag_features = torch.matmul(gated_attention, inst_features)
        print(bag_features.shape)

        bag_probability = self.cls(bag_features)

        return bag_probability, gated_attention


class WeakRMLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(13, 16, kernel_size=(1, 15), padding=(0, 7)),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d((1, 2))

        self.lstm = nn.LSTM(input_size=16, hidden_size=16, bidirectional=True, batch_first=True)

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
        # inputs = torch.squeeze(inputs, 0)
        input_bag = inputs  # torch.Size([1, 13, 40, 5])
        inst_conv1 = self.conv1(input_bag)  # torch.Size([1, 16, 40, 5])
        print(inst_conv1.shape)
        inst_pool1 = self.pool1(inst_conv1)  # torch.Size([1, 16, 40, 5])
        inst_pool1 = torch.squeeze(inst_pool1, 0)  # torch.Size([16, 40, 2])
        print(inst_pool1.shape)

        embedding = self.lstm(inst_pool1)  # torch.Size([13, 16, 32])
        print(embedding.shape)

        attention_v = self.attention_v(embedding)
        attention_u = self.attention_v(embedding)

        # print(attention_u.shape, attention_v.shape)
        gated_attention = self.attention_weights(attention_u * attention_v).permute((1, 0))

        gated_attention = nn.Softmax()(gated_attention)  # torch.Size([1, 13])

        bag_features = torch.matmul(gated_attention, embedding)
        # print(embedding.shape, gated_attention.shape)
        bag_probability = self.cls(bag_features)

        return bag_probability, gated_attention


class WSCNN(nn.Module):

    def __init__(self, instance_len=40, merging='MAX', ):
        super(WSCNN, self).__init__()

        assert merging in ['MAX', 'AVG']

        self.conv1 = nn.Sequential(
            nn.Conv2d(13, 16, kernel_size=(1, 15), padding=(0, 7)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), padding=(0, 7)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), padding=(0, 7)),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)
        self.pool1 = nn.MaxPool2d((1, instance_len))

        if merging == 'MAX':
            self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
            # self.pool2 = nn.GlobalMaxPooling2d()
        elif merging == 'AVG':
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
            # self.pool2 = nn.GlobalAveragePooling2d()

    def forward(self, inputs, training=True, mask=None):
        x = self.conv1(inputs)
        print(x.shape)
        x = self.pool1(x)
        x = self.conv2(x)
        print(x.shape)
        if training:
            x = self.dropout(x, training=training)
        x = self.conv3(x)
        out = self.pool2(x)
        return out


if __name__ == '__main__':
    x = torch.rand((1, 13, 40, 4))
    encoder = WeakRM()
    # encoder = WeakRMLSTM()
    # encoder = WSCNN()
    summary(encoder, x)
