from __future__ import annotations

import numpy as np
import torch
from temporal_negative_edge_sampler import NegativeEdgeSampler

from ..training.evaluator import eval_one_epoch
from ..utils.misc import EarlyStopMonitor
from ..walks.temporal_walk_store import TemporalWalkStore


def train(args, model, dataset, splits, logger, get_checkpoint_path, best_model_path):
    """Training loop using TemporalWalkStore + temporal negative sampling."""

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_src, train_dst, train_ts, train_e_idx, _ = splits.train
    val_src, val_dst, val_ts, val_e_idx, _ = splits.val
    test_src, test_dst, test_ts, test_e_idx, _ = splits.test

    train_order = np.argsort(train_ts)
    train_src = train_src[train_order]
    train_dst = train_dst[train_order]
    train_ts = train_ts[train_order]
    train_e_idx = train_e_idx[train_order]

    val_order = np.argsort(val_ts)
    val_src = val_src[val_order]
    val_dst = val_dst[val_order]
    val_ts = val_ts[val_order]
    val_e_idx = val_e_idx[val_order]

    test_order = np.argsort(test_ts)
    test_src = test_src[test_order]
    test_dst = test_dst[test_order]
    test_ts = test_ts[test_order]
    test_e_idx = test_e_idx[test_order]

    num_train = len(train_src)
    walk_generator_batch_size = args.walk_generator_batch_size

    early_stopper = EarlyStopMonitor(higher_better=True, tolerance=args.tolerance)
    best_ap = 0.0

    for epoch in range(args.n_epoch):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        walk_store = TemporalWalkStore(args, device=device)
        train_neg_sampler = NegativeEdgeSampler(
            is_directed=False,
            num_negatives_per_positive=args.negs,
            historical_negative_percentage=0.5,
            seed=args.seed,
        )

        for b_start in range(0, num_train, walk_generator_batch_size):
            b_end = min(b_start + walk_generator_batch_size, num_train)

            b_src = train_src[b_start:b_end]
            b_dst = train_dst[b_start:b_end]
            b_ts = train_ts[b_start:b_end]
            b_eidx = train_e_idx[b_start:b_end]

            _ingest_edges(walk_store, b_src, b_dst, b_ts, b_eidx, dataset)
            walk_store.build()

            train_neg_sampler.add_batch(b_src, b_dst, b_ts)
            neg_out = train_neg_sampler.sample_negatives()
            neg_targets = np.asarray(neg_out["targets"]).reshape(len(b_src), args.negs)

            valid_rows = np.all(neg_targets != -1, axis=1)
            if not np.any(valid_rows):
                continue

            b_src_valid = b_src[valid_rows]
            b_dst_valid = b_dst[valid_rows]
            neg_targets_valid = neg_targets[valid_rows]

            n_edges = len(b_src_valid)
            perm = np.random.permutation(n_edges)

            for mb_start in range(0, n_edges, args.bs):
                mb_end = min(mb_start + args.bs, n_edges)
                mb_idx = perm[mb_start:mb_end]

                src_mb = b_src_valid[mb_idx]
                dst_mb = b_dst_valid[mb_idx]
                neg_mb = neg_targets_valid[mb_idx]

                src_walks = walk_store.get(src_mb)
                dst_walks = walk_store.get(dst_mb)
                neg_walks_list = [walk_store.get(neg_mb[:, i]) for i in range(neg_mb.shape[1])]

                optimizer.zero_grad()
                loss = model.contrast(src_walks, dst_walks, neg_walks_list)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        val_walk_store = TemporalWalkStore(args, device=device)
        _ingest_edges(val_walk_store, train_src, train_dst, train_ts, train_e_idx, dataset)
        val_walk_store.build()

        val_neg_sampler = NegativeEdgeSampler(
            is_directed=False,
            num_negatives_per_positive=1,
            historical_negative_percentage=0.5,
            seed=args.seed,
        )
        val_neg_sampler.add_batch(train_src, train_dst, train_ts)
        val_neg_sampler.sample_negatives()

        val_ap, val_auc = eval_one_epoch(
            model=model,
            walk_store=val_walk_store,
            neg_sampler=val_neg_sampler,
            src=val_src,
            dst=val_dst,
            ts=val_ts,
            val_e_idx_l=val_e_idx,
        )

        logger.info(
            f'Epoch {epoch:3d} | loss {avg_loss:.4f} | '
            f'val AP {val_ap:.4f} | val AUC {val_auc:.4f}'
        )

        torch.save(model.state_dict(), get_checkpoint_path(epoch))
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'  -> new best model (AP={val_ap:.4f})')

        if early_stopper.early_stop_check(val_ap):
            logger.info(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    test_walk_store = TemporalWalkStore(args, device=device)
    _ingest_edges(test_walk_store, train_src, train_dst, train_ts, train_e_idx, dataset)
    _ingest_edges(test_walk_store, val_src, val_dst, val_ts, val_e_idx, dataset)
    test_walk_store.build()

    test_neg_sampler = NegativeEdgeSampler(
        is_directed=False,
        num_negatives_per_positive=1,
        historical_negative_percentage=0.5,
        seed=args.seed,
    )
    test_neg_sampler.add_batch(train_src, train_dst, train_ts)
    test_neg_sampler.sample_negatives()
    test_neg_sampler.add_batch(val_src, val_dst, val_ts)
    test_neg_sampler.sample_negatives()

    test_ap, test_auc = eval_one_epoch(
        model=model,
        walk_store=test_walk_store,
        neg_sampler=test_neg_sampler,
        src=test_src,
        dst=test_dst,
        ts=test_ts,
        val_e_idx_l=test_e_idx,
    )

    logger.info(f'Test AP {test_ap:.4f} | Test AUC {test_auc:.4f}')
    return {'test_ap': test_ap, 'test_auc': test_auc}


def _ingest_edges(walk_store, src, dst, ts, e_idx, dataset):
    efeat = dataset.e_feat[e_idx] if dataset.e_feat is not None else None
    walk_store.add_edges(src, dst, ts, efeat)
