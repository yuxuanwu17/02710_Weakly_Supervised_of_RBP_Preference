import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
from torchsummaryX import summary


class WeakRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.inst_length = 40

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
        inputs = inputs.permute((0, 2, 1))  # torch.Size([13, 5, 40])
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


import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1)).T  # (samples * timesteps, input_size)
        print(x_reshape.shape)  # torch.Size([320, 13])
        x_reshape = torch.unsqueeze(x_reshape, 0)
        print(x_reshape.shape)

        y, _ = self.module(x_reshape)
        print("After the lstm layer", y.shape)

        # # We have to reshape Y
        # if self.batch_first:
        #     y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        # else:
        #     y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        print(y.shape)

        return y


class WeakRMLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(1, 15), padding=(0, 7)),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d((2, 1))

        self.lstm = TimeDistributed(nn.LSTM(input_size=320, hidden_size=16, bidirectional=True, batch_first=True))

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
        )

        self.cls = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # print(inputs.shape)  # torch.Size([1, 13, 40, 4])
        inputs = inputs.permute((0, 3, 2, 1))  # torch.Size([1, 4, 40, 13])
        input_bag = inputs  # torch.Size([1, 4, 40, 13])
        # print(input_bag.shape)

        inst_conv1 = self.conv1(input_bag)  # torch.Size([1, 16, 40, 13])
        # print(inst_conv1.shape)  # torch.Size([1, 16, 40, 13])

        inst_pool1 = self.pool1(inst_conv1)  # torch.Size([1, 16, 20, 13])
        # print("After the max pooling", inst_pool1.shape)

        # inst_pool1 = torch.squeeze(inst_pool1, 0)  # torch.Size([16, 20, 13])
        # print(inst_pool1.shape)

        # inst_pool1 = inst_pool1.permute((2, 1, 0))
        # print(inst_pool1.shape)

        embedding = self.lstm(inst_pool1)  # torch.Size([13, 16, 32])
        # print(embedding[0].shape)
        # print(embedding[1].shape)

        # squeeze
        embedding = torch.squeeze(embedding, 0)
        attention_v = self.attention_v(embedding)
        attention_u = self.attention_v(embedding)

        # print(attention_u.shape, attention_v.shape)
        gated_attention = self.attention_weights(attention_u * attention_v)
        gated_attention = gated_attention.permute(1, 0)

        gated_attention = nn.Softmax()(gated_attention)  # torch.Size([1, 13])

        bag_features = torch.matmul(gated_attention, embedding)
        # print(embedding.shape, gated_attention.shape)
        bag_probability = self.cls(bag_features)

        return bag_probability, gated_attention


class WSCNN(nn.Module):

    def __init__(self, instance_len=40, merging='MAX', training=True):
        super(WSCNN, self).__init__()

        assert merging in ['MAX', 'AVG']

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=(1, 15), padding=(0, 7)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)
        self.pool1 = nn.MaxPool2d((instance_len, 1))

        if merging == 'MAX':
            self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
            # self.pool2 = nn.GlobalMaxPooling2d()
        elif merging == 'AVG':
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
            # self.pool2 = nn.GlobalAveragePooling2d()

    def forward(self, inputs, training=True, mask=None):
        print(inputs.shape)
        inputs = inputs.permute((0, 3, 2, 1))
        x = self.conv1(inputs)
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = self.conv2(x)
        print("After cov2d", x.shape)
        if training:
            x = self.dropout(x)
        x = self.conv3(x)
        print(x.shape)
        out = self.pool2(x)
        print(out.shape)
        return out


if __name__ == '__main__':
    x = torch.rand((1, 13, 40, 4))
    # x = torch.rand((1, 17, 20, 4))
    # encoder = WeakRM()
    encoder = WeakRMLSTM()
    # # encoder = WSCNN()
    # summary(encoder, x)
    # x = torch.rand(1, 101, 5)
    # encoder = Baseline()
    summary(encoder, x)
