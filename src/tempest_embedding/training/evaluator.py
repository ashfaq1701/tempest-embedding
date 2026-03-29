from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

TEST_BATCH_SIZE = 32


def eval_one_epoch(model, neg_tgt, src, dst, ts, label, val_e_idx_l=None):
    """Evaluate link prediction using precomputed negatives.

    Args:
        model:  the trained NeurTWs model (in eval mode).
        neg_tgt: (N, num_negs) precomputed negative target node IDs.
                 Only the first column is used (one negative per positive).
        src, dst, ts, label: edge arrays for this split.
        val_e_idx_l: edge index array (optional, unused by model).
    """
    val_ap, val_auc = [], []

    with torch.no_grad():
        model.eval()
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None
            size = len(src_l_cut)
            dst_l_fake = neg_tgt[s_idx:e_idx, 0]
            pos_prob, neg_prob = model.inference(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)
            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)
