from __future__ import annotations

import torch

from ..data.loader import load_dataset, split_dataset
from ..models.neurtws import NeurTWs
from ..training.trainer import train
from ..utils.logging import set_up_logger
from ..utils.misc import get_args
from ..utils.random import set_random_seed


DATA_DIR = './data'


def main():
    args, sys_argv = get_args()
    logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

    set_random_seed(args.seed)
    logger.info(f'Using seed {args.seed}')

    # ---- Device ----
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    # ---- Data ----
    dataset = load_dataset(DATA_DIR, args.data, args.data_usage)
    splits = split_dataset(dataset, args.mode, seed=args.seed)
    logger.info(f'Dataset {args.data}: {dataset.max_idx} nodes, '
                f'{len(dataset.src)} edges, '
                f'n_feat={dataset.n_feat.shape[1]}, e_feat={dataset.e_feat.shape[1]}')
    logger.info(f'Train {len(splits.train[0])} | Val {len(splits.val[0])} | '
                f'Test {len(splits.test[0])}')

    # ---- Model ----
    model = NeurTWs(
        n_feat=dataset.n_feat,
        e_feat=dataset.e_feat,
        pos_dim=args.pos_dim,
        pos_enc=args.pos_enc,
        max_walk_len=args.max_walk_len,
        num_walks_per_node=args.num_walks_per_node,
        mutual=args.walk_mutual,
        dropout_p=args.drop_out,
        walk_linear_out=args.walk_linear_out,
        solver=args.solver,
        step_size=args.step_size,
        tau=args.tau,
        logger=logger,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_params:,} trainable parameters')

    # ---- Train ----
    results = train(args, model, dataset, splits, logger,
                    get_checkpoint_path, best_model_path)

    logger.info(f'Final results: {results}')


if __name__ == '__main__':
    main()
