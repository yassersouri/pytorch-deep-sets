import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from .datasets import MNISTSummation, MNIST_TRANSFORM
from .networks import InvariantModel, SmallMNISTCNNPhi, SmallRho


class SumOfDigits(object):
    def __init__(self, lr=1e-3, wd=5e-3):
        self.lr = lr
        self.wd = wd
        self.train_db = MNISTSummation(min_len=2, max_len=10, dataset_len=100000, train=True, transform=MNIST_TRANSFORM)
        self.test_db = MNISTSummation(min_len=5, max_len=50, dataset_len=100000, train=False, transform=MNIST_TRANSFORM)

        self.the_phi = SmallMNISTCNNPhi()
        self.the_rho = SmallRho(input_size=10, output_size=1)

        self.model = InvariantModel(phi=self.the_phi, rho=self.the_rho)
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.summary_writer = SummaryWriter(
            log_dir='/home/souri/temp/deepsets/exp-lr:%1.5f-wd:%1.4f/' % (self.lr, self.wd))

    def train_1_epoch(self, epoch_num: int = 0):
        self.model.train()
        for i in tqdm(range(len(self.train_db))):
            loss = self.train_1_item(i)
            self.summary_writer.add_scalar('train_loss', loss, i + len(self.train_db) * epoch_num)

    def train_1_item(self, item_number: int) -> float:
        x, target = self.train_db.__getitem__(item_number)
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()

        x, target = Variable(x), Variable(target)

        self.optimizer.zero_grad()
        pred = self.model.forward(x)
        the_loss = F.mse_loss(pred, target)

        the_loss.backward()
        self.optimizer.step()

        the_loss_tensor = the_loss.data
        if torch.cuda.is_available():
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def evaluate(self):
        self.model.eval()
        totals = [0] * 51
        corrects = [0] * 51

        for i in tqdm(range(len(self.test_db))):
            x, target = self.test_db.__getitem__(i)

            item_size = x.shape[0]

            if torch.cuda.is_available():
                x = x.cuda()

            pred = self.model.forward(Variable(x)).data

            if torch.cuda.is_available():
                pred = pred.cpu().numpy().flatten()

            pred = int(round(float(pred[0])))
            target = int(round(float(target.numpy()[0])))

            totals[item_size] += 1

            if pred == target:
                corrects[item_size] += 1

        totals = np.array(totals)
        corrects = np.array(corrects)

        print(corrects / totals)
