from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

TEST_BATCH_SIZE = 32

def eval_one_epoch(model, walk_store, neg_sampler, src, dst, ts, val_e_idx_l=None):
    val_ap, val_auc = [], []

    with torch.no_grad():
        model.eval()
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            if s_idx >= e_idx:
                continue

            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]  # needed for sampler

            size = len(src_l_cut)

            if neg_sampler is None:
                raise ValueError("neg_sampler must be provided")

            # Temporal negative sampling
            neg_sampler.add_batch(src_l_cut, dst_l_cut, ts_l_cut)
            neg_out = neg_sampler.sample_negatives()

            neg_targets = np.asarray(neg_out["targets"]).reshape(size, -1)
            neg_targets = neg_targets[:, 0]

            # Filter sentinel values
            valid_mask = neg_targets != -1
            if not np.any(valid_mask):
                continue

            src_valid = src_l_cut[valid_mask]
            dst_valid = dst_l_cut[valid_mask]
            neg_valid = neg_targets[valid_mask]

            # Fetch walks
            src_walks = walk_store.get(src_valid)
            dst_walks = walk_store.get(dst_valid)
            neg_walks = walk_store.get(neg_valid)

            # Inference
            pos_prob, neg_prob = model.inference(
                src_walks, dst_walks, neg_walks
            )

            pred_score = np.concatenate([
                pos_prob.cpu().numpy(),
                neg_prob.cpu().numpy()
            ])

            true_label = np.concatenate([
                np.ones(len(src_valid)),
                np.zeros(len(src_valid))
            ])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return float(np.mean(val_ap)), float(np.mean(val_auc))
