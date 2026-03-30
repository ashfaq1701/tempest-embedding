from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

TEST_BATCH_SIZE = 32


def eval_one_epoch(model, neg_sampler, src, dst, ts, val_e_idx_l=None):
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
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if val_e_idx_l is not None else None

            size = len(src_l_cut)

            if neg_sampler is None:
                raise ValueError("neg_sampler must be provided")

            neg_sampler.add_batch(src_l_cut, dst_l_cut, ts_l_cut)
            neg_out = neg_sampler.sample_negatives()

            # sampler returns flattened (B * K)
            neg_targets = np.asarray(neg_out["targets"]).reshape(size, -1)

            # we use 1 negative per positive for evaluation
            neg_targets = neg_targets[:, 0]

            # --------------------------------------------------
            # Filter invalid sentinel negatives (-1)
            # --------------------------------------------------
            valid_mask = neg_targets != -1

            if not np.any(valid_mask):
                continue

            src_valid = src_l_cut[valid_mask]
            dst_valid = dst_l_cut[valid_mask]
            neg_valid = neg_targets[valid_mask]
            ts_valid = ts_l_cut[valid_mask]
            e_valid = e_l_cut[valid_mask] if e_l_cut is not None else None

            # --------------------------------------------------
            # Inference
            # --------------------------------------------------
            pos_prob, neg_prob = model.inference(
                src_valid, dst_valid, neg_valid, ts_valid, e_valid
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

    return np.mean(val_ap), np.mean(val_auc)
