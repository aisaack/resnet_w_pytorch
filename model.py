import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(
        self,
        stage,
        block,
        cin,
        cout,
        stride,
        name = 'ResBlock'
    ):
        super().__init__()
        self.name = name + f'{stage}-{block}'
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(in_channels = cin,
                                out_channels = cout,
                                kernel_size = (3, 3),
                                stride = stride,
                                padding = 1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(in_channels = cout,
                                out_channels = cout,
                                kernel_size = (3, 3),
                                stride = (1, 1),
                                padding = 1)
        if stride != 1:
            self.avg = AvgPool2d(kernel_size = (2, 2),
                                stride = (2, 2),
                                padding = 0)
            self.id = Conv2d(in_channels = cin,
                            out_channels = cout,
                            kernel_size = (1, 1),
                            stride = (1, 1),
                            padding = 0)

    def forward(self, X):
        x = self.conv1(F.relu(self.bn1(X)))
        x = self.conv2(F.relu(self.bn2(x)))
        if self.stride != 1:
            X = self.id(self.avg(X))
        return x + X


class NetworkStem(nn.Module):
    def __init__(
        self,
        cout,
        name = 'NetworkStem'
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = 3,
                                out_channels = cout,
                                kernel_size = (3, 3),
                                stride = (1, 1),
                                padding = 1)

    def forward(self, X):
        return self.conv(X)


class NetworkHead(nn.Module):
    def __init__(
        self,
        cin,
        num_class,
        name = 'NetworkHead'
    ):
        super().__init__()
        self.fc = nn.Linear(in_features = cin,
                            out_features = num_class)

    def forward(self, X):
        return self.fc(F.avg_pool(X, X.size()[2:]))


class ResNet(nn.Module):
    def __init__(
        self,
        num_layer,
        num_class
    ):
        super().__init__()
        architecture = {'cout': [16, 32, 64],
                        'stride': [1, 2, 2]}
        if num_layer == 20:
            architecture.update({'blocks':[3, 3, 3]})
        elif num_layer == 32:
            architecture.update({'blocks':[5, 5, 5]})
        elif num_layer == 44:
            architecture.update({'blocks':[7, 7, 7]})
        elif num_layer == 56:
            architecture.update({'blocks':[9, 9, 9]})
        
        cin = 16
        self.layers = nn.Sequential()
       
        self.layers.append(NetworkStem(architecture.get('cout')[0]))
        for stage, (cout, stride, blocks) in enumerate(zip(architecture.get('cout'),
                                                            architecture.get('stride'),
                                                            architecture.get('blocks'))):
            for block in range(blocks):
                if block > 0:
                    stride = 1
                self.layers.append(ResBlock(stage, block, cin, cout, stride))
                cin = cout
        self.layers.append(NetworkHead(cin, num_class))

    def forward(self, X):
        x = X
        for layer in self.layers:
            x = layer(x)
        return x

    def debug(self, X):
        x = X
        for layer in self.layers:
            x = layer(x)
            print(layer.name, ':', x.size())
        return x


def resnet20(num_class):
    return ResNet(20, num_class)

def resnet32(num_class):
    return ResNet(32, num_class)

def resnet44(num_class):
    return ResNet(44, num_class)

def resnet56(num_class):
    return ResNet(56, num_class)

if __name__ == '__main__':
    import numpy as np
    model = resnet20(10)
    x = np.random.randn(3, 32, 32, 3)
    model.debug(x)
