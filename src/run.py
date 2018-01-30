import click
from deepsets.settings import RANDOM_SEED
import numpy as np
import torch
from deepsets.experiments import SumOfDigits


@click.command()
@click.option('--random-seed', envvar='SEED', default=RANDOM_SEED)
def main(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    the_experiment = SumOfDigits(lr=1e-3)

    the_experiment.train_1_epoch()


if __name__ == '__main__':
    main()
