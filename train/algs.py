import logging
from abc import ABCMeta, abstractmethod

import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader

from nn.modules.losses import build_loss
from metrics.basic import AverageMeter, compute_accuracy
from utils import save_model

logger = logging.getLogger(__name__)

OPTIM = {
    'adam': optim.Adam,
    'amsgrad': optim.Adam,
    'sgd': optim.SGD
}


class AlgBase(metaclass=ABCMeta):
    def __init__(self, trainer, datamap, args):
        self.trainer = trainer
        self.datamap = datamap
        self.args = args
        self.round = 0

    def optim(self, net):
        args = self.args
        optimargs = {
            'params': filter(lambda p: p.requires_grad, net.parameters()),
            'lr': args.lr,
            'weight_decay': args.reg
        }
        if args.optimizer in ('adam', 'amsgrad'):
            return OPTIM[args.optimizer](**optimargs, amsgrad=args.optimizer == 'amsgrad')
        elif args.optimizer == 'sgd':
            return OPTIM[args.optimizer](**optimargs, momentum=args.momentum)

    @abstractmethod
    def train_single_local(self, net_idx, trainloader, testloader):
        pass

    @abstractmethod
    def train_selected_locals(self, selected):
        pass

    @abstractmethod
    def aggregate_weights(self, selected):
        pass


class FedAvg(AlgBase):
    def __init__(self, trainer, datamap, args):
        super().__init__(trainer, datamap, args)
        self.criterion = build_loss(args.loss)

    def train_single_local(self, net_idx, trainloader, testloader):
        args = self.args
        device = torch.device(args.device)
        net = self.trainer.local_models[net_idx]
        optimizer = self.optim(net)

        metrics = {
            'total_loss': AverageMeter(),
            args.loss: AverageMeter(),
            'train_acc': AverageMeter(),
        }

        for epoch in range(1, args.epochs + 1):
            metrics['total_loss'].reset()
            metrics[args.loss].reset()
            metrics['train_acc'].reset()
            for data, target in trainloader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                data.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(data)
                loss, additional = self.criterion(out, target, model=net, decay=args.odecay)
                loss.backward()
                optimizer.step()
                # Metrics update
                acc = sum(target == out.cpu().clone().detach().argmax(1)).item() / target.size(0)
                metrics['total_loss'].update(loss, len(data))
                metrics[args.loss].update(additional if additional else loss, len(data))
                metrics['train_acc'].update(acc, len(data))

            # Logging
            logger.info(f'Epoch: {epoch:>3} | Loss: {metrics["total_loss"].avg:.6f}')
            wandb.log(
                data={
                    f'Client {net_idx}': {
                        'train': {
                            'Loss': metrics['total_loss'].avg,
                            args.loss: metrics[args.loss].avg,
                            'train_acc': metrics['train_acc'].avg
                        },
                    },
                    'epochsum': (self.round - 1) * args.epochs + epoch
                }
            )

            # Save local model
            cond_comm = (self.round % args.save_round == 0) or self.round == args.comm_round
            cond_epoch = (epoch % args.save_epoch == 0) or epoch == args.epochs
            if args.save_local and cond_comm and cond_epoch:
                save_model(net, args.name, args.modeldir, f'comm{self.round:03}-epoch{epoch:03}-CLIENT{net_idx:02}')

        train_acc = compute_accuracy(net, trainloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, testloader, get_confusion_matrix=True, device=device)
        logger.info(f'>> Train accuracy: {train_acc * 100:5.2f} %')
        logger.info(f'>>  Test accuracy: {test_acc * 100:5.2f} %')
        wandb.log(
            data={
                f'Client {net_idx}': {
                    'train': {'Accuracy': train_acc},
                    'test': {'Accuracy': test_acc},
                },
                'round': self.round,
                'epochsum': self.round * args.epochs
            },
        )
        return train_acc, test_acc

    def train_selected_locals(self, selected):
        self.round += 1
        avg_local_test_acc = 0.0

        for net_id, net in self.nets.items():
            if net_id not in selected:
                continue
            data_indices = self.datamap[net_id]

            logger.info('-' * 58)
            logger.info(f'Training client {net_id:>3} with {len(data_indices):>6} data')

            loader = DataLoader(
                self.trainer.trainsets[net_id], self.args.batch_size, shuffle=True, drop_last=False,
                pin_memory=True, num_workers=self.args.num_workers)
            testloader = DataLoader(
                self.trainer.testsets[net_id], 128, shuffle=False, drop_last=False,
                pin_memory=True, num_workers=self.args.num_workers)
            trainacc, testacc = self.train_single_local(net_id, loader, testloader)

            avg_local_test_acc += testacc

        logger.info('-' * 58)
        avg_local_test_acc /= len(selected)
        logger.info(f'average local test acc {avg_local_test_acc}')

        nets_list = list(self.nets.values())
        return nets_list

    def aggregate_weights(self, selected):
        # TODO
        # Fix to use einsum with partial(functools)
        # Model weight synchronization
        global_params = self.trainer.global_model.state_dict()
        if self.round == 1:
            if self.args.is_same_initial:
                for idx in selected:
                    self.trainer.local_models[idx].load_state_dict(global_params)
        else:
            for idx in selected:
                self.trainer.local_models[idx].load_state_dict(global_params)

        # Update global model
        total_data_points = sum([len(self.datamap[r]) for r in selected])
        fed_avg_freqs = [len(self.datamap[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_para = self.trainer.local_models[selected[idx]].state_dict()
            if idx == 0:
                for key in net_para:
                    global_params[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_params[key] += net_para[key] * fed_avg_freqs[idx]
        self.trainer.global_model.load_state_dict(global_params)


class FedProx(AlgBase):
    def __init__(self, trainer, datamap, args):
        super().__init__(trainer, datamap, args)
        self.mu = args.mu
        self.criterion = build_loss(args.loss)

    def _proximal(self, net):
        global_weights = list(self.trainer.global_model.parameters())
        reg = 0.0
        for param_index, param in enumerate(net.parameters()):
            reg += ((self.mu / 2) * torch.norm((param - global_weights[param_index])) ** 2)
        return reg

    def train_single_local(self, net_idx, trainloader, testloader):
        args = self.args
        device = torch.device(args.device)
        net = self.trainer.local_models[net_idx]
        optimizer = self.optim(net)

        metrics = {
            'total_loss': AverageMeter(),
            args.loss: AverageMeter(),
        }

        for epoch in range(1, args.epochs + 1):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(trainloader):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = self.criterion(out, target)

                # Proximal loss term for FedProx
                loss += self._proximal(net)

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        train_acc = compute_accuracy(net, trainloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, testloader, get_confusion_matrix=True, device=device)

        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)

        logger.info(' ** Training complete **')
        return train_acc, test_acc

    def train_selected_locals(self, selected):
        pass

    def aggregate_weights(self, selected):
        pass


class SCAFFOLD(AlgBase):
    def __init__(self, trainer, datamap, args):
        super().__init__(trainer, datamap, args)
        self.round = 0
        self.criterion = build_loss(args.loss)

    def train_single_local(self, net_idx, trainloader, testloader):
        pass

    def train_selected_locals(self, selected):
        pass

    def aggregate_weights(self):
        pass


class FedNova(AlgBase):
    def __init__(self, trainer, datamap, args):
        super().__init__(trainer, datamap, args)
        self.round = 0
        self.criterion = build_loss(args.loss)

    def train_single_local(self, net_idx, trainloader, testloader):
        pass

    def train_selected_locals(self, selected):
        pass

    def aggregate_weights(self, selected):
        pass

class MOON(AlgBase):
    def __init__(self, trainer, datamap, args):
        super().__init__(trainer, datamap, args)
        self.round = 0
        self.criterion = build_loss(args.loss)

    def train_single_local(self, net_idx, trainloader, testloader):
        pass

    def train_selected_locals(self, selected):
        pass

    def aggregate_weights(self, selected):
        pass