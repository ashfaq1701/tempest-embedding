from __future__ import annotations

import logging
import os
import time

from .misc import process_sampling_numbers


def set_up_logger(args, sys_argv):
    n_degree, n_layer = process_sampling_numbers(args.n_degree, args.n_layer)
    n_degree = [str(n) for n in n_degree]
    runtime_id = '{}-{}-{}-{}-{}-{}'.format(
        str(time.time()), args.data, args.mode[0], n_layer, 'k'.join(n_degree), args.pos_dim
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    os.makedirs('log', exist_ok=True)
    file_path = f'log/{runtime_id}.log'
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f'Create log file at {file_path}')
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)

    checkpoint_root = './saved_checkpoints/'
    best_model_root = './best_models/'
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(best_model_root, exist_ok=True)

    checkpoint_dir = os.path.join(checkpoint_root, runtime_id)
    best_model_dir = os.path.join(best_model_root, runtime_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    logger.info(f'Create checkpoint directory {checkpoint_dir}')
    logger.info(f'Create best model directory {best_model_dir}')

    get_checkpoint_path = lambda epoch: os.path.join(checkpoint_dir, f'checkpoint-epoch-{epoch}.pth')
    best_model_path = os.path.join(best_model_dir, 'best-model.pth')
    return logger, get_checkpoint_path, best_model_path
