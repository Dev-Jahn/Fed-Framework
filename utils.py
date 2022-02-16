import os
import logging

import numpy as np

from data.partition import partition_data
from models.cnns import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta
    )
    return net_dataidx_map
