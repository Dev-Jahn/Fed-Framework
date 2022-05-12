import logging

import wandb
from torch import optim

from data.dataloader import get_dataloader
from losses import build_loss
from metrics.basic import AverageMeter, compute_accuracy
from utils import save_model

logger = logging.getLogger(__name__)


def train_local(net_id, net, trainloader, testloader, comm_round, args, device):
    # train_acc = compute_accuracy(net, trainloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, testloader, get_confusion_matrix=True, device=device)
    # logger.info(f'<< Train accuracy: {train_acc * 100:5.2f} %')
    # logger.info(f'<<  Test accuracy: {test_acc * 100:5.2f} %')
    # wandb.log(
    #     data={
    #         f'Client {net_id}': {
    #             'train': {'Accuracy': train_acc},
    #             'test': {'Accuracy': test_acc},
    #         },
    #         'round': comm_round - 0.5
    #     },
    # )

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.reg)
    criterion = build_loss(args.loss)

    metrics = {
        'total_loss': AverageMeter(),
        args.loss: AverageMeter(),
    }
    for epoch in range(1, args.epochs + 1):
        metrics['total_loss'].reset()
        metrics[args.loss].reset()
        for batch_idx, (x, target) in enumerate(trainloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss, additional = criterion(out, target, model=net, decay=args.odecay)
            loss.backward()
            optimizer.step()
            # Metrics update
            metrics['total_loss'].update(loss, len(x))
            metrics[args.loss].update(additional if additional else loss, len(x))

        # Logging
        logger.info(f'Epoch: {epoch:>3} | Loss: {metrics["total_loss"].avg:.6f}')
        wandb.log(
            data={
                f'Client {net_id}': {
                    'train': {
                        'Loss': metrics['total_loss'].avg,
                        args.loss: metrics[args.loss].avg
                    },
                },
                'epochsum': (comm_round - 1) * args.epochs + epoch
            }
        )

        # Save local model
        cond_comm = (comm_round % args.save_round == 0) or comm_round == args.comm_round
        cond_epoch = (epoch % args.save_epoch == 0) or epoch == args.epochs
        if args.save_local and cond_comm and cond_epoch:
            save_model(net, args.name, args.modeldir, f'comm{comm_round:03}-epoch{epoch:03}-CLIENT{net_id:02}')

        # calc acc for local (optional)
        # train_acc = compute_accuracy(net, train_dataloader, device=device)
        # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, trainloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, testloader, get_confusion_matrix=True, device=device)
    logger.info(f'>> Train accuracy: {train_acc * 100:5.2f} %')
    logger.info(f'>>  Test accuracy: {test_acc * 100:5.2f} %')
    wandb.log(
        data={
            f'Client {net_id}': {
                'train': {'Accuracy': train_acc},
                'test': {'Accuracy': test_acc},
            },
            'round': comm_round
        },
    )
    return train_acc, test_acc


def train_nets(nets, selected, args, net_dataidx_map, loaders, comm_round, testloader=None, device='cuda'):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info('-' * 58)
        logger.info(f'Training client {net_id:>3} with {len(dataidxs):>6} data')

        net.to(device)
        trainacc, testacc = train_local(net_id, net, loaders[net_id], testloader, comm_round, args, device=device)
        net.cpu()

        avg_acc += testacc
        # Save model
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    logger.info('-' * 58)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list
