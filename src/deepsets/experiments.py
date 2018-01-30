import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from datasets import MNISTSummation, MNIST_TRANSFORM
from networks import InvariantModel, SmallMNISTCNNPhi, SmallRho
from tqdm import tqdm
from tensorboardX import SummaryWriter


class SumOfDigits(object):
    def __init__(self, lr=1e-3):
        self.lr = lr
        self.train_db = MNISTSummation(min_len=2, max_len=10, dataset_len=10000, train=True, transform=MNIST_TRANSFORM)
        self.test_db = MNISTSummation(min_len=5, max_len=50, dataset_len=10000, train=False, transform=MNIST_TRANSFORM)

        self.the_phi = SmallMNISTCNNPhi()
        self.the_rho = SmallRho(input_size=10, output_size=1)

        self.model = InvariantModel(phi=self.the_phi, rho=self.the_rho)
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.summary_writer = SummaryWriter(log_dir='/home/souri/temp/deepsets/exp1/')

    def train_1_epoch(self):
        for i in tqdm(range(len(self.train_db))):
            self.train_1_item(i)

    def train_1_item(self, item_number: int) -> float:
        x, target = self.train_db.__getitem__(item_number)
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()

        x, target = Variable(x), Variable(target)

        pred = self.model.forward(x)
        the_loss = F.mse_loss(pred, target)

        the_loss.backward()
        self.optimizer.step()

        the_loss_tensor = the_loss.data
        if torch.cuda.is_available():
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        self.summary_writer.add_scalar('train_loss', the_loss_float, item_number)

        return the_loss_float
