from __future__ import annotations

import time

import numpy as np
import torch

from temporal_negative_edge_sampler import NegativeEdgeSampler

from ..training.evaluator import eval_one_epoch
from ..utils.misc import EarlyStopMonitor
from ..walks.batching import WalkBatcher
from ..walks.tempest import TempestWalkBackend


def train(args, model, dataset, splits, logger, get_checkpoint_path, best_model_path):
    """Training loop with temporal negative sampling for train/val/test."""

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Unpack & sort training edges by timestamp
    # ------------------------------------------------------------------
    train_src, train_dst, train_ts, train_e_idx, _ = splits.train
    sort_idx = np.argsort(train_ts)
    train_src = train_src[sort_idx]
    train_dst = train_dst[sort_idx]
    train_ts = train_ts[sort_idx]
    train_e_idx = train_e_idx[sort_idx]

    val_src, val_dst, val_ts, val_e_idx, _ = splits.val
    test_src, test_dst, test_ts, test_e_idx, _ = splits.test

    num_train = len(train_src)
    walk_generator_batch_size = args.walk_generator_batch_size

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

        # Fresh walk backend and fresh temporal negative sampler per epoch
        backend = TempestWalkBackend(args)
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

            # Ingest current batch into graph state
            _ingest_edges(backend, b_src, b_dst, b_ts, b_eidx, dataset)

            # Generate walks for current graph state
            nodes, times, lens, edge_feats = backend.generate_walks()
            nodes, times, lens, edge_feats = batcher.reshape_walks(
                nodes, times, lens, edge_feats,
            )
            model.set_walks(nodes, times, lens, edge_feats)

            # Temporal negatives for this batch
            train_neg_sampler.add_batch(b_src, b_dst, b_ts)
            neg_out = train_neg_sampler.sample_negatives()

            neg_targets = np.asarray(neg_out["targets"]).reshape(len(b_src), args.negs)

            # Drop rows containing invalid sentinel negatives (-1)
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

                optimizer.zero_grad()
                loss = model.contrast(src_mb, dst_mb, neg_mb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # ----------------------------------------------------------
        # Validation
        # Current walk state after training epoch corresponds to full train graph
        # Build a temporal negative sampler on train history, then evaluate val
        # ----------------------------------------------------------
        val_neg_sampler = NegativeEdgeSampler(
            is_directed=False,
            num_negatives_per_positive=1,
            historical_negative_percentage=0.5,
            seed=args.seed,
        )
        val_neg_sampler.add_batch(train_src, train_dst, train_ts)
        val_neg_sampler.sample_negatives()  # commit train edges to sampler history

        def sample_val_neg(size: int) -> np.ndarray:
            # Consume val edges sequentially in chunks aligned with evaluator calls
            raise RuntimeError("sample_val_neg should be chunk-driven; use eval_with_temporal_sampler")

        val_ap, val_auc = eval_with_temporal_sampler(
            model=model,
            src=val_src,
            dst=val_dst,
            ts=val_ts,
            e_idx=val_e_idx,
            sampler=val_neg_sampler,
        )

        logger.info(
            f'Epoch {epoch:3d} | loss {avg_loss:.4f} | '
            f'val AP {val_ap:.4f} | val AUC {val_auc:.4f} | '
            f'time {time.time() - t0:.1f}s'
        )

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

    # Graph state for test = train + val
    test_backend = TempestWalkBackend(args)
    _ingest_edges(test_backend, train_src, train_dst, train_ts, train_e_idx, dataset)
    _ingest_edges(test_backend, val_src, val_dst, val_ts, val_e_idx, dataset)

    nodes, times, lens, edge_feats = test_backend.generate_walks()
    nodes, times, lens, edge_feats = batcher.reshape_walks(
        nodes, times, lens, edge_feats,
    )
    model.set_walks(nodes, times, lens, edge_feats)

    test_neg_sampler = NegativeEdgeSampler(
        is_directed=False,
        num_negatives_per_positive=1,
        historical_negative_percentage=0.5,
        seed=args.seed,
    )
    test_neg_sampler.add_batch(train_src, train_dst, train_ts)
    test_neg_sampler.sample_negatives()  # commit train
    test_neg_sampler.add_batch(val_src, val_dst, val_ts)
    test_neg_sampler.sample_negatives()  # commit val

    test_ap, test_auc = eval_with_temporal_sampler(
        model=model,
        src=test_src,
        dst=test_dst,
        ts=test_ts,
        e_idx=test_e_idx,
        sampler=test_neg_sampler,
    )

    logger.info(f'Test AP {test_ap:.4f} | Test AUC {test_auc:.4f}')

    results = {
        'test_ap': test_ap,
        'test_auc': test_auc,
    }

    return results


def eval_with_temporal_sampler(model, src, dst, ts, e_idx, sampler):
    """Evaluate sequentially using the temporal negative sampler.

    Assumes sampler already contains all prior history before this split.
    """
    import math
    from sklearn.metrics import average_precision_score, roc_auc_score

    test_batch_size = 32
    aps, aucs = [], []

    with torch.no_grad():
        model.eval()
        num_instances = len(src)
        num_batches = math.ceil(num_instances / test_batch_size)

        for k in range(num_batches):
            s_idx = k * test_batch_size
            e_idx_batch = min(num_instances, s_idx + test_batch_size)
            if s_idx >= e_idx_batch:
                continue

            src_cut = src[s_idx:e_idx_batch]
            dst_cut = dst[s_idx:e_idx_batch]
            ts_cut = ts[s_idx:e_idx_batch]
            edge_idx_cut = e_idx[s_idx:e_idx_batch] if e_idx is not None else None

            sampler.add_batch(src_cut, dst_cut, ts_cut)
            neg_out = sampler.sample_negatives()

            neg_tgt = np.asarray(neg_out["targets"]).reshape(len(src_cut), 1).squeeze(1)

            # keep only rows with a valid sampled negative
            valid = neg_tgt != -1
            if not np.any(valid):
                continue

            src_eval = src_cut[valid]
            dst_eval = dst_cut[valid]
            ts_eval = ts_cut[valid]
            edge_idx_eval = edge_idx_cut[valid] if edge_idx_cut is not None else None
            neg_eval = neg_tgt[valid]

            pos_prob, neg_prob = model.inference(
                src_eval, dst_eval, neg_eval, ts_eval, edge_idx_eval
            )

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(len(src_eval)), np.zeros(len(src_eval))])

            aps.append(average_precision_score(true_label, pred_score))
            aucs.append(roc_auc_score(true_label, pred_score))

    return float(np.mean(aps)), float(np.mean(aucs))


def _ingest_edges(backend, src, dst, ts, e_idx, dataset):
    """Add edges (with features) into Tempest backend."""
    efeat = dataset.e_feat[e_idx] if dataset.e_feat is not None else None
    backend.add_edges(src, dst, ts, efeat)
