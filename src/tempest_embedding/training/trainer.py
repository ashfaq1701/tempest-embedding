from __future__ import annotations

import time

import numpy as np
import torch
from temporal_negative_edge_sampler import collect_all_negatives_by_timestamp

from ..training.evaluator import eval_one_epoch
from ..utils.misc import EarlyStopMonitor
from ..walks.batching import WalkBatcher
from ..walks.tempest import TempestWalkBackend


def _precompute_negatives(src_arrays, dst_arrays, ts_arrays, num_negs):
    """Precompute random negatives for edges across multiple splits.

    Concatenates all splits chronologically, calls the temporal negative edge
    sampler once, then splits results back.  Returns a list of (N_i, num_negs)
    arrays — one per input split.
    """
    lengths = [len(s) for s in src_arrays]
    all_src = np.concatenate(src_arrays)
    all_dst = np.concatenate(dst_arrays)
    all_ts = np.concatenate(ts_arrays)

    # Sort globally by timestamp so the sampler sees edges in order
    order = np.argsort(all_ts, kind='stable')
    inv = np.argsort(order, kind='stable')  # to unsort results

    _, neg_tgt = collect_all_negatives_by_timestamp(
        all_src[order].astype(np.int32),
        all_dst[order].astype(np.int32),
        all_ts[order].astype(np.int64),
        is_directed=False,
        num_negatives_per_positive=num_negs,
        historical_negative_percentage=0.0,
    )

    # (total * num_negs,) → (total, num_negs), then restore original order
    neg_tgt = neg_tgt.reshape(-1, num_negs)[inv]

    # Split back per input array
    parts, offset = [], 0
    for n in lengths:
        parts.append(neg_tgt[offset:offset + n])
        offset += n
    return parts


def train(args, model, dataset, splits, logger, get_checkpoint_path, best_model_path):
    """Training loop with incremental Tempest ingestion.

    Per epoch:
      1. Create a fresh TempestWalkBackend.
      2. For each chronological batch of --walk_generator_batch_size edges:
           a. Ingest the batch's edges (cumulative, no eviction).
           b. Generate walks for all nodes in current graph state.
           c. Mini-batch over the batch's edges for gradient updates.
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
    walk_generator_batch_size = args.walk_generator_batch_size

    val_src, val_dst, val_ts, val_e_idx, val_label = splits.val
    test_src, test_dst, test_ts, test_e_idx, test_label = splits.test

    # ------------------------------------------------------------------
    # Precompute negatives for train / val / test
    # ------------------------------------------------------------------
    train_neg_tgt, val_neg_tgt, test_neg_tgt = _precompute_negatives(
        [train_src, val_src, test_src],
        [train_dst, val_dst, test_dst],
        [train_ts, val_ts, test_ts],
        num_negs=args.negs,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

        for b_start in range(0, num_train, walk_generator_batch_size):
            b_end = min(b_start + walk_generator_batch_size, num_train)

            b_src = train_src[b_start:b_end]
            b_dst = train_dst[b_start:b_end]
            b_ts = train_ts[b_start:b_end]
            b_eidx = train_e_idx[b_start:b_end]
            b_neg = train_neg_tgt[b_start:b_end]  # (batch, negs)

            # Ingest + walk generation per batch
            _ingest_edges(backend, b_src, b_dst, b_ts, b_eidx, dataset)

            nodes, times, lens, edge_feats = backend.generate_walks()
            nodes, times, lens, edge_feats = batcher.reshape_walks(
                nodes, times, lens, edge_feats,
            )
            model.set_walks(nodes, times, lens, edge_feats)

            # Mini-batch training over this batch's edges
            n_edges = b_end - b_start
            perm = np.random.permutation(n_edges)

            for mb_start in range(0, n_edges, args.bs):
                mb_end = min(mb_start + args.bs, n_edges)
                mb_idx = perm[mb_start:mb_end]

                src_mb = b_src[mb_idx]
                dst_mb = b_dst[mb_idx]
                neg_mb = b_neg[mb_idx]  # (mb_size, negs)

                optimizer.zero_grad()
                loss = model.contrast(src_mb, dst_mb, neg_mb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # ----------------------------------------------------------
        # Validation (walks are from the last batch = full train graph)
        # ----------------------------------------------------------
        val_ap, val_auc = eval_one_epoch(
            model, val_neg_tgt, val_src, val_dst, val_ts, val_label, val_e_idx,
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

    # Graph state for test: train + val edges
    test_backend = TempestWalkBackend(args)
    _ingest_edges(test_backend, train_src, train_dst, train_ts, train_e_idx, dataset)
    _ingest_edges(test_backend, val_src, val_dst, val_ts, val_e_idx, dataset)

    nodes, times, lens, edge_feats = test_backend.generate_walks()
    nodes, times, lens, edge_feats = batcher.reshape_walks(
        nodes, times, lens, edge_feats,
    )
    model.set_walks(nodes, times, lens, edge_feats)

    test_ap, test_auc = eval_one_epoch(
        model, test_neg_tgt, test_src, test_dst, test_ts, test_label, test_e_idx,
    )
    logger.info(f'Test AP {test_ap:.4f} | Test AUC {test_auc:.4f}')

    results = {'test_ap': test_ap, 'test_auc': test_auc}

    # Inductive sub-splits (if present)
    if splits.test_new_new is not None:
        nn_src, nn_dst, nn_ts, nn_eidx, nn_label = splits.test_new_new
        if len(nn_src) > 0:
            [nn_neg_tgt] = _precompute_negatives(
                [nn_src], [nn_dst], [nn_ts], num_negs=args.negs,
            )
            nn_ap, nn_auc = eval_one_epoch(
                model, nn_neg_tgt, nn_src, nn_dst, nn_ts, nn_label, nn_eidx,
            )
            logger.info(f'Test new-new  AP {nn_ap:.4f} | AUC {nn_auc:.4f}')
            results['nn_ap'], results['nn_auc'] = nn_ap, nn_auc

    if splits.test_new_old is not None:
        no_src, no_dst, no_ts, no_eidx, no_label = splits.test_new_old
        if len(no_src) > 0:
            [no_neg_tgt] = _precompute_negatives(
                [no_src], [no_dst], [no_ts], num_negs=args.negs,
            )
            no_ap, no_auc = eval_one_epoch(
                model, no_neg_tgt, no_src, no_dst, no_ts, no_label, no_eidx,
            )
            logger.info(f'Test new-old  AP {no_ap:.4f} | AUC {no_auc:.4f}')
            results['no_ap'], results['no_auc'] = no_ap, no_auc

    return results


def _ingest_edges(backend, src, dst, ts, e_idx, dataset):
    """Add a set of edges (with their features) into the Tempest backend."""
    efeat = dataset.e_feat[e_idx] if dataset.e_feat is not None else None
    backend.add_edges(src, dst, ts, efeat)
