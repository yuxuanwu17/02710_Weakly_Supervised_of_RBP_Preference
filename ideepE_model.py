class CNN(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=200, stride=(1, 1), padding=0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        out1_size = (window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        maxpool_size = (out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size=(1, 10), stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out2_size = (maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        maxpool2_size = (out2_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1
        self.drop1 = nn.Dropout(p=0.25)
        print
        'maxpool_size', maxpool_size
        self.fc1 = nn.Linear(maxpool2_size * nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


class CNN_LSTM(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=200, stride=(1, 1), padding=0, num_layers=2):
        super(CNN_LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        out1_size = (window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        maxpool_size = (out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1
        self.downsample = nn.Conv2d(nb_filter, 1, kernel_size=(1, 10), stride=stride, padding=padding)
        input_size = (maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        self.layer2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.downsample(out)
        out = torch.squeeze(out, 1)
        # pdb.set_trace()
        if cuda:
            x = x.cuda()
            h0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size)).cuda()
            c0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size))
        out, _ = self.layer2(out, (h0, c0))
        out = out[:, -1, :]
        # pdb.set_trace()
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


def convR(in_channels, out_channels, kernel_size, stride=1, padding=(0, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, stride=stride, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, nb_filter=16, kernel_size=(1, 3), stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = convR(in_channel, nb_filter, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convR(nb_filter, nb_filter, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(nb_filter)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, nb_filter=16, channel=7, labcounts=12, window_size=36, kernel_size=(1, 3),
                 pool_size=(1, 3), num_classes=2, hidden_size=200):
        super(ResNet, self).__init__()
        self.in_channels = channel
        self.conv = convR(self.in_channels, nb_filter, kernel_size=(4, 10))
        cnn1_size = window_size - 7
        self.bn = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, nb_filter, layers[0], kernel_size=kernel_size)
        self.layer2 = self.make_layer(block, nb_filter * 2, layers[1], 1, kernel_size=kernel_size,
                                      in_channels=nb_filter)
        self.layer3 = self.make_layer(block, nb_filter * 4, layers[2], 1, kernel_size=kernel_size,
                                      in_channels=2 * nb_filter)
        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_1_size = (cnn1_size - (pool_size[1] - 1) - 1) / pool_size[1] + 1
        last_layer_size = 4 * nb_filter * avgpool2_1_size
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1, kernel_size=(1, 10), in_channels=16):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                convR(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, downsample=downsample))
        # self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print x.data.cpu().numpy().shape
        # x = x.view(x.size(0), 1, x.size(1), x.size(2))
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        # pdb.set_trace()
        # print self.layer2
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        # pdb.set_trace()
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):
    def __init__(self, labcounts=4, window_size=107, channel=7, growth_rate=6, block_config=(16, 16, 16),
                 compression=0.5,
                 num_init_features=12, bn_size=2, drop_rate=0, avgpool_size=(1, 8),
                 num_classes=2):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(channel, num_init_features, kernel_size=(4, 10), stride=1, bias=False)),
        ]))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        last_size = window_size - 7
        # Each denseblock
        num_features = num_init_features
        # last_size =  window_size
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                last_size = (last_size - (2 - 1) - 1) / 2 + 1
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        avgpool2_1_size = (last_size - (self.avgpool_size[1] - 1) - 1) / self.avgpool_size[1] + 1
        num_features = num_features * avgpool2_1_size
        print
        num_features
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x = x.view(x.size(0), 1, x.size(1), x.size(2))
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        out = F.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.features[0](x)
        out = self.features[1](out)
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


def get_all_data(protein, channel=7):
    data = load_graphprot_data(protein)
    test_data = load_graphprot_data(protein, train=False)
    # pdb.set_trace()
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data)
        test_bags, true_y = get_bag_data_1_channel(test_data)
    else:
        train_bags, label = get_bag_data(data)
        # pdb.set_trace()
        test_bags, true_y = get_bag_data(test_data)

    return train_bags, label, test_bags, true_y


def run_network(model_type, X_train, test_bags, y_train, channel=7, window_size=107):
    print
    'model training for ', model_type
    # nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter=16, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter=16, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter=16, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size=window_size, channel=channel, labcounts=4)
    else:
        print
        'only support CNN, CNN-LSTM, ResNet and DenseNet model'

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=100, nb_epoch=50)

    print
    'predicting'
    pred = model.predict_proba(test_bags)
    return pred, model