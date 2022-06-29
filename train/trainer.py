import logging

import numpy as np
import torch
import wandb
from torch.utils.data import ConcatDataset, DataLoader

from data.dataloader import get_dataset
from data.partition import partition_data
from metrics.basic import compute_accuracy
from models.nets import init_nets
from train.algs import FedAvg, FedProx, SCAFFOLD, FedNova
from utils import save_model

ALGS = {
    'fedavg': FedAvg,
    'fedprox': FedProx,
    'scaffold': SCAFFOLD,
    'fednova': FedNova
}
logger = logging.getLogger(__name__)


class FederatedTrainer:
    def __init__(self, alg, args):
        if alg not in ALGS.keys():
            raise NotImplementedError(f'Unsupported training algorithm "{alg}"')
        self.alg = ALGS[alg]()
        self.args = args

        self.local_models = None
        self.global_model = None
        self.trainloader_global = None
        self.testloader_global = None
        self.datamap = None
        self.trainsets = []
        self.testsets = []
        self.data_cls_counts = 0

    def prepare(self):
        args = self.args
        # Data partitioning
        logger.info('Partitioning data...')
        self.datamap, self.data_cls_counts = partition_data(
            args.dataset, args.datadir, args.logdir, args.partition, args.n_clients, beta=args.beta, seed=args.seed
        )

        # Prepare dataset
        self.trainsets = [get_dataset(args.dataset, args.datadir, self.datamap, i, True, args) for i in range(args.n_clients)]
        self.testsets = [get_dataset(args.dataset, args.datadir, self.datamap, i, False, args) for i in range(args.n_clients)]

        # Add noise to global
        if args.noise > 0:
            trainset_global = ConcatDataset(self.trainsets)
            testset_global = ConcatDataset(self.testsets)
        else:
            trainset_global = get_dataset(args.dataset, args.datadir, None, None, train=True, args=args)
            testset_global = get_dataset(args.dataset, args.datadir, None, None, train=False, args=args)

        logger.info(f'Global train size: {len(trainset_global)}')
        logger.info(f'Global test  size: {len(testset_global)}')

        self.trainloader_global = DataLoader(
            trainset_global, args.batch_size, shuffle=True, drop_last=False,
            pin_memory=True, num_workers=args.num_workers,
        )
        self.testloader_global = DataLoader(
            testset_global, 128, shuffle=False, drop_last=False,
            pin_memory=True, num_workers=args.num_workers,
            persistent_workers=True
        )

    def train(self):
        args = self.args
        device = torch.device(args.device)

        logger.info('Initializing nets...')
        self.local_models, local_model_meta_data, layer_type = init_nets(args.dropout, args.n_clients, args)
        logger.info('Complete.')
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        self.global_model = global_models[0]

        # commented out for consecutive run
        # check_disk_space(
        #     global_model, args.n_clients, args.comm_round, args.save_round,
        #     args.save_local, args.epochs, args.save_epoch
        # )

        for round_ in range(1, args.comm_round + 1):
            logger.info('=' * 58)
            logger.info('Communication round: ' + str(round_))
            arr = np.arange(args.n_clients)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_clients * args.sample)]

            # Model weight synchronization
            global_params = self.global_model.state_dict()
            if round_ == 1:
                if args.is_same_initial:
                    for idx in selected:
                        self.local_models[idx].load_state_dict(global_params)
            else:
                for idx in selected:
                    self.local_models[idx].load_state_dict(global_params)

            self.alg.train_selected_locals(selected)
            self.alg.aggregate_weights(selected)

            # Test global model
            self.global_model.to(device)
            train_acc = compute_accuracy(self.global_model, self.trainloader_global, device=device)
            test_acc, conf_matrix = compute_accuracy(
                self.global_model, self.testloader_global, get_confusion_matrix=True, device=device
            )
            # global_model.cpu()

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
                save_model(self.global_model, args.name, args.modeldir, f'comm{round_:03}-GLOBAL')
        logger.info('=' * 58)
        #####################################################################################
        #####################################################################################

        if args.alg == 'fedprox':
            logger.info('Initializing nets')
            nets, local_model_meta_data, layer_type = init_nets(args.dropout, args.n_clients, args)
            global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
            global_model = global_models[0]

            global_params = global_model.state_dict()

            if args.is_same_initial:
                for net_id, net in nets.items():
                    net.load_state_dict(global_params)

            for round_ in range(args.comm_round):
                logger.info('Communication round: ' + str(round_))

                arr = np.arange(args.n_clients)
                np.random.shuffle(arr)
                selected = arr[:int(args.n_clients * args.sample)]

                global_params = global_model.state_dict()
                if round_ == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_params)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_params)

                local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, testargs=testargs_global,
                                        device=device)
                global_model.to('cpu')

                # update global model
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                for idx in range(len(selected)):
                    net_para = nets[selected[idx]].state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_params[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_params[key] += net_para[key] * fed_avg_freqs[idx]
                global_model.load_state_dict(global_params)
                trainloader_global = DataLoader(**trainargs_global)
                logger.info('global n_training: %d' % len(trainloader_global))

                train_acc = compute_accuracy(global_model, trainloader_global, device=device)
                del trainloader_global
                testloader_global = DataLoader(**testargs_global)
                logger.info('global n_test: %d' % len(testloader_global))
                test_acc, conf_matrix = compute_accuracy(
                    global_model, testloader_global, get_confusion_matrix=True, device=device
                )
                del testloader_global

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

            global_params = global_model.state_dict()
            if args.is_same_initial:
                for net_id, net in nets.items():
                    net.load_state_dict(global_params)

            for round_ in range(args.comm_round):
                logger.info('Communication round: ' + str(round_))

                arr = np.arange(args.n_clients)
                np.random.shuffle(arr)
                selected = arr[:int(args.n_clients * args.sample)]

                global_params = global_model.state_dict()
                if round_ == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_params)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_params)

                local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,
                                         testargs=testargs_global, device=device)

                # update global model
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                for idx in range(len(selected)):
                    net_para = nets[selected[idx]].state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_params[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_params[key] += net_para[key] * fed_avg_freqs[idx]
                global_model.load_state_dict(global_params)

                global_model.to('cpu')
                trainloader_global = DataLoader(**trainargs_global)
                logger.info('global n_training: %d' % len(trainloader_global))
                train_acc = compute_accuracy(global_model, trainloader_global, device=device)
                del trainloader_global
                testloader_global = DataLoader(**testargs_global)
                logger.info('global n_test: %d' % len(testloader_global))
                test_acc, conf_matrix = compute_accuracy(
                    global_model, testloader_global, get_confusion_matrix=True, device=device
                )
                del testloader_global

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
                data_sum += len(self.data_cls_counts[i])
            portion = []
            for i in range(args.n_clients):
                portion.append(len(self.data_cls_counts[i]) / data_sum)

            global_params = global_model.state_dict()
            if args.is_same_initial:
                for net_id, net in nets.items():
                    net.load_state_dict(global_params)

            for round_ in range(args.comm_round):
                logger.info('Communication round: ' + str(round_))

                arr = np.arange(args.n_clients)
                np.random.shuffle(arr)
                selected = arr[:int(args.n_clients * args.sample)]

                global_params = global_model.state_dict()
                if round_ == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_params)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_params)

                _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map,
                                                                    testargs=testargs_global, device=device)
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

                global_model.to('cpu')
                trainloader_global = DataLoader(**trainargs_global)
                logger.info('global n_training: %d' % len(trainloader_global))
                train_acc = compute_accuracy(global_model, trainloader_global, device=device)
                del trainloader_global
                testloader_global = DataLoader(**testargs_global)
                logger.info('global n_test: %d' % len(testloader_global))
                test_acc, conf_matrix = compute_accuracy(
                    global_model, testloader_global, get_confusion_matrix=True, device=device
                )
                del testloader_global

                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

        elif args.alg == 'local_training':
            logger.info('Initializing nets')
            nets, local_model_meta_data, layer_type = init_nets(args.dropout, args.n_clients, args)
            arr = np.arange(args.n_clients)
            train_nets(nets, arr, args, net_dataidx_map, testargs=testargs_global, device=device)
        else:
            raise NotImplementedError()
