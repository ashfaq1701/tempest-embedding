from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..training.evaluator import eval_one_epoch
from ..training.negative import RandEdgeSampler
from ..utils.misc import EarlyStopMonitor
from ..walks.batching import WalkBatcher
from ..walks.tempest import TempestWalkBackend


def train(args, model, dataset, splits, logger, get_checkpoint_path, best_model_path):
    """Time-windowed training loop with incremental Tempest ingestion.

    Per epoch:
      1. Create a fresh TempestWalkBackend.
      2. For each time window (edges sorted chronologically):
           a. Ingest the window's edges (cumulative, no eviction).
           b. Generate walks for all nodes in current graph state.
           c. Mini-batch over the window's edges for gradient updates.
      3. Validate via eval_one_epoch; checkpoint + early stopping.

    After training: load best model, build train+val graph, evaluate on test set.

    Returns: dict with test_ap, test_auc (and optionally nn_ap/auc, no_ap/auc
             for inductive splits).
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Unpack & sort training edges by timestamp
    # ------------------------------------------------------------------
    train_src, train_dst, train_ts, train_e_idx, train_label = splits.train
    sort_idx = np.argsort(train_ts)
    train_src = train_src[sort_idx]
    train_dst = train_dst[sort_idx]
    train_ts = train_ts[sort_idx]
    train_e_idx = train_e_idx[sort_idx]

    num_train = len(train_src)
    window_size = args.temporal_micro_batch_max_size
    num_windows = math.ceil(num_train / window_size)

    val_src, val_dst, val_ts, val_e_idx, val_label = splits.val

    # ------------------------------------------------------------------
    # Samplers & helpers
    # ------------------------------------------------------------------
    train_sampler = RandEdgeSampler([train_src], [train_dst])
    val_sampler = RandEdgeSampler([train_src, val_src], [train_dst, val_dst])
    batcher = WalkBatcher(args.num_walks_per_node, args.max_walk_len)
    early_stopper = EarlyStopMonitor(higher_better=True, tolerance=args.tolerance)

    best_ap = 0.0

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(args.n_epoch):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        # Fresh walk backend each epoch (walks are stochastic)
        backend = TempestWalkBackend(args)

        for w in range(num_windows):
            w_start = w * window_size
            w_end = min(w_start + window_size, num_train)

            # --- Incremental ingestion ---
            w_src = train_src[w_start:w_end]
            w_dst = train_dst[w_start:w_end]
            w_ts = train_ts[w_start:w_end]
            w_eidx = train_e_idx[w_start:w_end]

            _ingest_edges(backend, w_src, w_dst, w_ts, w_eidx, dataset)

            # --- Walk generation ---
            nodes, times, lens, edge_feats = backend.generate_walks()
            nodes, times, lens, edge_feats = batcher.reshape_walks(
                nodes, times, lens, edge_feats,
            )
            model.set_walks(nodes, times, lens, edge_feats)

            # --- Mini-batch training over this window ---
            n_edges = w_end - w_start
            perm = np.random.permutation(n_edges)

            for b_start in range(0, n_edges, args.bs):
                b_end = min(b_start + args.bs, n_edges)
                b_idx = perm[b_start:b_end]

                src_b = w_src[b_idx]
                dst_b = w_dst[b_idx]

                # Negative destinations: (batch, num_negs)
                neg_b = np.stack(
                    [train_sampler.sample(len(src_b))[1] for _ in range(args.negs)],
                    axis=1,
                )

                optimizer.zero_grad()
                loss = model.contrast(src_b, dst_b, neg_b)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # ----------------------------------------------------------
        # Validation (walks are from the last window = full train graph)
        # ----------------------------------------------------------
        val_ap, val_auc = eval_one_epoch(
            model, val_sampler, val_src, val_dst, val_ts, val_label, val_e_idx,
        )

        logger.info(
            f'Epoch {epoch:3d} | loss {avg_loss:.4f} | '
            f'val AP {val_ap:.4f} | val AUC {val_auc:.4f} | '
            f'time {time.time() - t0:.1f}s'
        )

        # Checkpoint
        torch.save(model.state_dict(), get_checkpoint_path(epoch))
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'  -> new best model (AP={val_ap:.4f})')

        if early_stopper.early_stop_check(val_ap):
            logger.info(f'Early stopping at epoch {epoch}')
            break

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    test_src, test_dst, test_ts, test_e_idx, test_label = splits.test

    # Graph state for test: train + val edges
    test_backend = TempestWalkBackend(args)
    _ingest_edges(test_backend, train_src, train_dst, train_ts, train_e_idx, dataset)
    _ingest_edges(test_backend, val_src, val_dst, val_ts, val_e_idx, dataset)

    nodes, times, lens, edge_feats = test_backend.generate_walks()
    nodes, times, lens, edge_feats = batcher.reshape_walks(
        nodes, times, lens, edge_feats,
    )
    model.set_walks(nodes, times, lens, edge_feats)

    test_sampler = RandEdgeSampler(
        [train_src, val_src, test_src],
        [train_dst, val_dst, test_dst],
    )
    test_ap, test_auc = eval_one_epoch(
        model, test_sampler, test_src, test_dst, test_ts, test_label, test_e_idx,
    )
    logger.info(f'Test AP {test_ap:.4f} | Test AUC {test_auc:.4f}')

    results = {'test_ap': test_ap, 'test_auc': test_auc}

    # Inductive sub-splits (if present)
    if splits.test_new_new is not None:
        nn_src, nn_dst, nn_ts, nn_eidx, nn_label = splits.test_new_new
        if len(nn_src) > 0:
            nn_sampler = RandEdgeSampler([nn_src], [nn_dst])
            nn_ap, nn_auc = eval_one_epoch(
                model, nn_sampler, nn_src, nn_dst, nn_ts, nn_label, nn_eidx,
            )
            logger.info(f'Test new-new  AP {nn_ap:.4f} | AUC {nn_auc:.4f}')
            results['nn_ap'], results['nn_auc'] = nn_ap, nn_auc

    if splits.test_new_old is not None:
        no_src, no_dst, no_ts, no_eidx, no_label = splits.test_new_old
        if len(no_src) > 0:
            no_sampler = RandEdgeSampler([no_src], [no_dst])
            no_ap, no_auc = eval_one_epoch(
                model, no_sampler, no_src, no_dst, no_ts, no_label, no_eidx,
            )
            logger.info(f'Test new-old  AP {no_ap:.4f} | AUC {no_auc:.4f}')
            results['no_ap'], results['no_auc'] = no_ap, no_auc

    return results


def _ingest_edges(backend, src, dst, ts, e_idx, dataset):
    """Add a set of edges (with their features) into the Tempest backend."""
    efeat = dataset.e_feat[e_idx] if dataset.e_feat is not None else None
    backend.add_edges(src, dst, ts, efeat)
