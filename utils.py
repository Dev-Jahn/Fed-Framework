import os
import sys
from datetime import datetime
import logging
from contextlib import redirect_stdout
import io
import re
import math

from torchsummary import summary

from models.cnns import *


def init_logger(name, logdir):
    log_path = f'{name}_log-{datetime.now().strftime("%Y-%m-%d-%H:%M-%S")}.log'
    formatter = logging.Formatter(
        fmt='%(asctime)s|%(levelname)-8s| %(message)s',
        datefmt='%m-%d %H:%M'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_debug_handler = logging.FileHandler(os.path.join(logdir, log_path), mode='w')
    file_debug_handler.setLevel(logging.DEBUG)
    file_debug_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_debug_handler]
    )


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def save_model(model, exprname, modeldir, postfix):
    logger = logging.getLogger(__name__)
    tstamp = datetime.now().strftime('%y%m%d-%H-%M-%S')
    path = os.path.join(modeldir, f'{tstamp}_{exprname}_{postfix}.pt')
    logger.info(f'Saving {path}')
    with open(path, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(model, model_index, device="cpu"):
    logger = logging.getLogger(__name__)
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


def calc_total_param_size(model, n_clients, comm_round, save_round, save_local, epoch, save_epoch):
    units = ['MB', 'GB', 'TB']
    f = io.StringIO()
    with redirect_stdout(f):
        summary(model, (3, 32, 32), device='cpu')
    s = f.getvalue()
    # size in MB
    size = float(re.search('(?<=Params size \(MB\): )[0-9.]+', s)[0])

    total = 0
    # Global models
    total += (math.ceil(comm_round / save_round)) * size
    # Local models
    if save_local:
        total += comm_round * (math.ceil(epoch / save_epoch)) * size * n_clients
    unitidx = 0
    while int(total) > 1024:
        total /= 1024
        unitidx += 1
    return f'{str(round(total, 2))} {units[unitidx]}'


def check_disk_space(model, n_clients, comm_round, save_round, save_local, epoch, save_epoch):
    # Check disk space before train
    psize = calc_total_param_size(
        model, n_clients, comm_round, save_round,
        save_local, epoch, save_epoch
    )
    print('#'*80)
    print(f'You need {psize} of disk space.')
    if not check_yes_or_no('Do you want to proceed?'):
        exit(0)


def check_yes_or_no(question):
    while True:
        ans = input(f'{question} (y/n) > ')
        if ans in ['y', 'Y']:
            return True
        elif ans in ['n', 'N']:
            return False
