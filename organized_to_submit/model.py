import torch
import torch.nn as nn

class WeakRM(nn.Module):
    """
    used for channel = 4, AGCT
    """

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
        )

        self.cls = nn.Sequential(
            nn.Linear(320, 2),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax()

    def forward(self, inputs, training=True, mask=None):
        inputs = torch.squeeze(inputs, 0)
        inputs = inputs.permute((0, 2, 1))  # torch.Size([13, 5, 40])
        inst_conv1 = self.inst_conv1(inputs)  # torch.Size([13, 32, 20])
        inst_conv1 = self.dropout(inst_conv1)
        inst_conv2 = self.inst_conv2(inst_conv1)

        inst_features = nn.Flatten()(inst_conv2)

        attention_v = self.attention_v(inst_features)
        attention_u = self.attention_v(inst_features)

        # print(attention_u*attention_v)
        # print(self.attention_weights(attention_u * attention_v))

        gated_attention = self.attention_weights(attention_u * attention_v).permute((1, 0))
        # print(gated_attention)

        gated_attention = self.softmax(gated_attention)  # torch.Size([1, 13])

        bag_features = torch.matmul(gated_attention, inst_features)

        bag_probability = self.cls(bag_features)

        return bag_probability, gated_attention
    
class WeakRMwithStructure(nn.Module):
    """
    used for channel = 4, AGCT
    """

    def __init__(self, training=True):
        super().__init__()

        self.inst_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=15, padding=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.structure_conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=15, padding=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.attention_v = nn.Sequential(
            nn.Linear(640, 128),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(640, 128),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.cls = nn.Sequential(
            nn.Linear(640, 2),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax()

    def forward(self, inputs, structures, training=True, mask=None):
        inputs = torch.squeeze(inputs, 0)
        inputs = inputs.permute((0, 2, 1)) 
        
        structures = torch.squeeze(structures, 0)
        structures = structures.permute((0, 2, 1)) 
        # print(structures.shape)
        inst_features = self.inst_conv(inputs) 
        structure_features = self.structure_conv(structures)
        
        # print(inst_features.shape, structure_features.shape)
        inst_features = torch.cat((inst_features, structure_features), dim = 1)
        # print(inst_features.shape)
        attention_v = self.attention_v(inst_features)
        attention_u = self.attention_v(inst_features)

        # print(attention_u*attention_v)
        # print(self.attention_weights(attention_u * attention_v))

        gated_attention = self.attention_weights(attention_u * attention_v).permute((1, 0))
        # print(gated_attention)

        gated_attention = self.softmax(gated_attention)  # torch.Size([1, 13])

        bag_features = torch.matmul(gated_attention, inst_features)

        bag_probability = self.cls(bag_features)

        return bag_probability, gated_attention



class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1)).T 
        x_reshape = torch.unsqueeze(x_reshape, 0)
        y, _ = self.module(x_reshape)
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
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = inputs.permute((0, 3, 2, 1))  # torch.Size([1, 4, 40, 13])
        input_bag = inputs  # torch.Size([1, 4, 40, 13])

        inst_conv1 = self.conv1(input_bag)  # torch.Size([1, 16, 40, 13])

        inst_pool1 = self.pool1(inst_conv1)  # torch.Size([1, 16, 20, 13])

        embedding = self.lstm(inst_pool1)  # torch.Size([13, 16, 32])

        # squeeze
        embedding = torch.squeeze(embedding, 0)

        attention_v = self.attention_v(embedding)
        attention_u = self.attention_v(embedding)

        gated_attention = self.attention_weights(attention_u * attention_v)
        gated_attention = gated_attention.permute(1, 0)

        gated_attention = nn.Softmax()(gated_attention)  # torch.Size([1, 13])

        bag_features = torch.matmul(gated_attention, embedding)

        bag_probability = self.cls(bag_features)

        return bag_probability, gated_attention

class WSCNN(nn.Module):
    def __init__(self, instance_len=40, merging='MAX', training=True):
        super(WSCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(1, 15), padding=(0, 7)),
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
        elif merging == 'AVG':
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.cls = nn.Sequential(
            nn.Linear(1, 2),
            nn.Sigmoid()
        )

    def forward(self, inputs, training=True, mask=None):
        inputs = inputs.permute((0, 3, 2, 1))
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        out = self.pool2(x)
        out = self.cls(out)
        return out[0][0], None