from typing import Any
import torch.nn as nn
from torch.autograd import Variable
from torch import FloatTensor
import torch
import torch.nn.functional as F

NetIO = Any[FloatTensor, Variable]


class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: NetIO) -> NetIO:
        # compute the representation for each data point
        x = self.phi.forward(x)

        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        x = torch.sum(x, dim=0, keepdim=True)

        # compute the output
        out = self.rho.forward(x)

        return out


class SmallMNISTCNNPhi(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(F.max_pool2d(x), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        return x


class SmallRho(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, self.output_size)

    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
