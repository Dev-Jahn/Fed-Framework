import json
import argparse
import datetime
import logging
import os

import numpy as np
import torch
import wandb

from train.trainer import FederatedTrainer
from utils import mkdirs, init_logger

DATASETS = ['mnist', 'fmnist', 'cifar10', 'svhn', 'celeba', 'femnist', 'generated', 'rcv1', 'SUSY', 'covtype', 'a9a']


def get_args():
    parser = argparse.ArgumentParser()
    # Experiment setting
    parser.add_argument('--name', type=str, required=True, help='Name of each experiment')

    # Model & Dataset
    parser.add_argument('--arch', type=str, default='MLP', help='Neural network architecture used in training')
    parser.add_argument('--modeldir', type=str, required=False, default='./ckpt/', help='Model directory path')
    parser.add_argument('--dataset', type=str, choices=DATASETS, help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default='./data/', help='Data directory')
    parser.add_argument('--save_round', type=int, default=10, help='Save model once in n comm rounds')
    parser.add_argument('--save_local', action='store_true', help='Save local model for analysis')
    parser.add_argument('--save_epoch', type=int, default=5, help='Save local model once in n epochs')
    # Train, Hyperparams, Optimizer
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--dropout', type=float, required=False, default=0.3, help='Dropout probability. Default=0.0')
    parser.add_argument('--loss', type=str, choices=['ce', 'srip', 'ocnn'], default='ce', help='Loss function')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'amsgrad', 'sgd'], default='sgd', help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--nesterov', type=bool, default=True, help='nesterov momentum')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--reg', type=float, default=1e-5, help='L2 losses strength')
    parser.add_argument('--odecay', type=float, default=1e-2, help='Orth loss strength')
    parser.add_argument('--augment', action='store_true', help='Data augmentation')
    parser.add_argument('--desync-bn', action='store_true', help='If True, does not aggregate BatchNorm weights')
    # Averaging algorithms
    parser.add_argument('--alg', type=str, choices=['fedavg', 'fedprox', 'scaffold', 'fednova'],
                        help='communication strategy')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication rounds')
    parser.add_argument('--is_same_initial', type=int, default=1, help='All models with same parameters in fedavg')
    # Data partitioning
    parser.add_argument('--n_clients', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--partition', type=str,
                        choices=['homo', 'noniid-labeldir', 'iid-diff-quantity', 'mixed', 'real', 'femnist'] \
                                + ['noniid-#label' + str(i) for i in range(10)],
                        default='homo', help='the data partitioning strategy')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, choices=['space', 'level'], default='level',
                        help='Different level of noise or different space of noise')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    # Misc.
    parser.add_argument('--num_workers', default=8, type=int, help='core usage (default: 1)')
    parser.add_argument('--ngpu', default=1, type=int, help='total number of gpus (default: 1)')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--amp', action='store_true', help='Turn Automatic Mixed Precision on')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--logdir', type=str, required=False, default='./logs/', help='Log directory path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # Logging
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    args_path = f'{args.name}_arguments-{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")}.json'
    with open(os.path.join(args.logdir, args_path), 'w') as f:
        json.dump(str(args), f)
    init_logger(args.name, args.logdir)

    logger = logging.getLogger()

    # Wandb
    wandb.init(
        name=args.name,
        config=args.__dict__,
        project='federated-learning',
        tags=['train', args.arch, args.dataset, args.loss],
    )

    logger.info(f'Device: {args.device}')

    # Set random seeds
    logger.info(f'Seed: {args.seed}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = FederatedTrainer(args.alg, args)
    trainer.prepare()
    trainer.train()

    wandb.finish()
