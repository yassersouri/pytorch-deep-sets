import click
from deepsets.settings import RANDOM_SEED
import numpy as np
import torch


@click.command()
@click.option('--random-seed', envvar='SEED', default=RANDOM_SEED)
def main(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


if __name__ == '__main__':
    main()
