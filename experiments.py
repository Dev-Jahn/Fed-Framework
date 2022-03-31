import json
import argparse
import copy
import datetime
import logging
import os
import sys

import numpy as np
import torch
import wandb
from torch.utils.data import ConcatDataset, DataLoader

from algs.fedavg import train_nets
from algs.fednova import local_train_net_fednova
from algs.fedprox import local_train_net_fedprox
from algs.scaffold import local_train_net_scaffold
from data.dataloader import get_dataloader
from data.partition import partition_data
from metrics.basic import compute_accuracy
from models.nets import init_nets
from utils import mkdirs, init_logger, check_disk_space, save_model

DATASETS = ['mnist', 'fmnist', 'cifar10', 'svhn', 'celeba', 'femnist', 'generated', 'rcv1', 'SUSY', 'covtype', 'a9a']


def get_args():
    parser = argparse.ArgumentParser()
    # Experiment setting
    parser.add_argument('--name', type=str, required=True, help='Name of each experiment')

    # Model & Dataset
    parser.add_argument('--model', type=str, default='MLP', help='neural network used in training')
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
    parser.add_argument('--dropout', type=float, required=False, default=0.0, help='Dropout probability. Default=0.0')
    parser.add_argument('--loss', type=str, choices=['ce', 'orth'], default='ce', help='Loss function')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'amsgrad', 'sgd'], default='sgd', help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--nesterov', type=bool, default=True, help='nesterov momentum')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--reg', type=float, default=1e-5, help='L2 losses strength')
    parser.add_argument('--odecay', type=float, default=1e-2, help='Orth loss strength')
    # Averaging algorithms
    parser.add_argument('--alg', type=str, choices=['fedavg', 'fedprox', 'scaffold', 'fednova'],
                        help='communication strategy')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
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
    parser.add_argument('--ngpu', default=1, type=int, help='total number of gpus (default: 1)')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--init_seed', type=int, default=0, help='Random seed')
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
        tags=['train', args.model, args.dataset, args.loss],
    )

    # Device
    device = torch.device(args.device)
    logger.info(f'Device: {device}')

    # Set random seeds
    logger.info(f'Seed: {args.init_seed}')
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data partitioning
    logger.info('Partitioning data...')
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_clients, beta=args.beta)

    # Prepare dataloader
    logger.info('Creating dataloaders...')
    if args.noise_type == 'space':
        noise_level = lambda net_idx: 0 if net_idx == args.n_clients - 1 else args.noise
        dl_args = lambda net_idx: {'net_id': net_idx, 'total': args.n_clients - 1}
    else:
        noise_level = lambda net_idx: args.noise / (args.n_clients - 1) * net_idx
        dl_args = lambda net_idx: {}
    trainloaders, testloaders, trainsets, testsets = [
        {idx: obj for idx, obj in enumerate(tup)} for tup in list(zip(
            *[get_dataloader(
                args.dataset, args.datadir, args.batch_size, 32,
                net_dataidx_map[i], noise_level(i), **dl_args(i)
            ) for i in range(args.n_clients)]
        ))
    ]
    # if noise
    if args.noise > 0:
        trainloader_global = DataLoader(dataset=ConcatDataset(trainsets), batch_size=args.batch_size, shuffle=True)
        testloader_global = DataLoader(dataset=ConcatDataset(testsets), batch_size=32, shuffle=False)
    else:
        trainloader_global, testloader_global, trainset_global, testset_global = get_dataloader(
            args.dataset, args.datadir, args.batch_size, 32
        )
    logger.info(f'Global train size: {len(trainset_global)}')
    logger.info(f'Global test  size: {len(testset_global)}')

    if args.alg == 'fedavg':
        logger.info('Initializing nets...')
        nets, local_model_meta_data, layer_type = init_nets(args.dropout, args.n_clients, args)
        logger.info('Complete.')
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        global_model = global_models[0]

        check_disk_space(
            global_model, args.n_clients, args.comm_round, args.save_round,
            args.save_local, args.epochs, args.save_epoch
        )

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round_ in range(1, args.comm_round + 1):
            logger.info('=' * 58)
            logger.info('Communication round: ' + str(round_))
            arr = np.arange(args.n_clients)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_clients * args.sample)]

            global_para = global_model.state_dict()
            if round_ == 1:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            train_nets(nets, selected, args, net_dataidx_map, trainloaders, round_, testloader=testloader_global,
                       device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            global_model.to(device)
            train_acc = compute_accuracy(global_model, trainloader_global, device=device)
            test_acc, conf_matrix = compute_accuracy(
                global_model, testloader_global, get_confusion_matrix=True, device=device
            )
            global_model.cpu()

            logger.info(f'>> Global Train accuracy: {train_acc * 100:5.2f} %')
            logger.info(f'>> Global  Test accuracy: {test_acc * 100:5.2f} %')
            wandb.log(
                data={
                    f'Global': {
                        'train': {'Accuracy': train_acc},
                        'test': {'Accuracy': test_acc},
                    },
                    'round': round_
                },
            )

            # Save global model
            if (round_ % args.save_round == 0) or round_ == args.comm_round:
                save_model(global_model, args.name, args.modeldir, f'comm{round_:03}-GLOBAL')
        logger.info('=' * 58)

    elif args.alg == 'fedprox':
        logger.info('Initializing nets')
        nets, local_model_meta_data, layer_type = init_nets(args.dropout, args.n_clients, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round_ in range(args.comm_round):
            logger.info('Communication round: ' + str(round_))

            arr = np.arange(args.n_clients)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_clients * args.sample)]

            global_para = global_model.state_dict()
            if round_ == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=testloader_global,
                                    device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(trainloader_global))
            logger.info('global n_test: %d' % len(testloader_global))

            train_acc = compute_accuracy(global_model, trainloader_global, device=device)
            test_acc, conf_matrix = compute_accuracy(
                global_model, testloader_global, get_confusion_matrix=True, device=device
            )

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info('Initializing nets')
        nets, local_model_meta_data, layer_type = init_nets(args.dropout, args.n_clients, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.dropout, args.n_clients, args)
        c_globals, _, _ = init_nets(0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round_ in range(args.comm_round):
            logger.info('Communication round: ' + str(round_))

            arr = np.arange(args.n_clients)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_clients * args.sample)]

            global_para = global_model.state_dict()
            if round_ == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,
                                     test_dl=testloader_global, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(trainloader_global))
            logger.info('global n_test: %d' % len(testloader_global))

            global_model.to('cpu')
            train_acc = compute_accuracy(global_model, trainloader_global, device=device)
            test_acc, conf_matrix = compute_accuracy(
                global_model, testloader_global, get_confusion_matrix=True, device=device
            )

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info('Initializing nets')
        nets, local_model_meta_data, layer_type = init_nets(args.dropout, args.n_clients, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_clients)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_clients):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_clients):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_clients):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round_ in range(args.comm_round):
            logger.info('Communication round: ' + str(round_))

            arr = np.arange(args.n_clients)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_clients * args.sample)]

            global_para = global_model.state_dict()
            if round_ == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map,
                                                                test_dl=testloader_global, device=device)
            total_n = sum(n_list)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i] / total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)

            logger.info('global n_training: %d' % len(trainloader_global))
            logger.info('global n_test: %d' % len(testloader_global))

            global_model.to('cpu')
            train_acc = compute_accuracy(global_model, trainloader_global, device=device)
            test_acc, conf_matrix = compute_accuracy(
                global_model, testloader_global, get_confusion_matrix=True, device=device
            )

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'local_training':
        logger.info('Initializing nets')
        nets, local_model_meta_data, layer_type = init_nets(args.dropout, args.n_clients, args)
        arr = np.arange(args.n_clients)
        train_nets(nets, arr, args, net_dataidx_map, testloader=testloader_global, device=device)
    else:
        raise NotImplementedError()

    wandb.finish()
