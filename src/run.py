import click
import numpy as np
import torch

from deepsets.experiments import SumOfDigits
from deepsets.settings import RANDOM_SEED


@click.command()
@click.option('--random-seed', envvar='SEED', default=RANDOM_SEED)
def main(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    the_experiment = SumOfDigits(lr=1e-3)

    for i in range(20):
        the_experiment.train_1_epoch(i)
        the_experiment.evaluate()


if __name__ == '__main__':
    main()
