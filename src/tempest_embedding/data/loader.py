from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math
import random
import numpy as np
import pandas as pd


@dataclass
class TemporalDataset:
    g_df: pd.DataFrame
    src: np.ndarray
    dst: np.ndarray
    e_idx: np.ndarray
    label: np.ndarray
    ts: np.ndarray
    n_feat: np.ndarray
    e_feat: np.ndarray
    max_idx: int


@dataclass
class DataSplits:
    train: tuple
    val: tuple
    test: tuple
    test_new_new: tuple | None = None
    test_new_old: tuple | None = None
    val_time: float | None = None
    test_time: float | None = None


def load_dataset(data_dir: str | Path, dataset_name: str, data_usage: float = 1.0) -> TemporalDataset:
    data_dir = Path(data_dir)
    g_df = pd.read_csv(data_dir / f'ml_{dataset_name}.csv')
    if data_usage < 1:
        g_df = g_df.iloc[:int(data_usage * g_df.shape[0])]
    e_feat = np.load(data_dir / f'ml_{dataset_name}.npy')
    n_feat = np.load(data_dir / f'ml_{dataset_name}_node.npy')
    src = g_df.u.values
    dst = g_df.i.values
    e_idx = g_df.idx.values
    label = g_df.label.values
    ts = g_df.ts.values
    max_idx = max(src.max(), dst.max())
    assert np.unique(np.stack([src, dst])).shape[0] == max_idx or ~math.isclose(1, data_usage)
    assert n_feat.shape[0] == max_idx + 1 or ~math.isclose(1, data_usage)
    return TemporalDataset(g_df, src, dst, e_idx, label, ts, n_feat, e_feat, max_idx)


def split_dataset(ds: TemporalDataset, mode: str, seed: int = 0) -> DataSplits:
    src_l, dst_l, ts_l, e_idx_l, label_l = ds.src, ds.dst, ds.ts, ds.e_idx, ds.label
    val_time, test_time = list(np.quantile(ds.g_df.ts, [0.70, 0.85]))

    if mode == 't':
        valid_train_flag = ts_l <= val_time
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
        valid_test_flag = ts_l > test_time
        return DataSplits(
            train=(src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]),
            val=(src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]),
            test=(src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]),
            val_time=val_time,
            test_time=test_time,
        )

    assert mode == 'i'
    rng = random.Random(seed)
    total_node_set = set(np.unique(np.hstack([ds.g_df.u.values, ds.g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_pool = set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))
    mask_node_set = set(rng.sample(list(mask_node_pool), int(0.1 * num_total_unique_nodes)))
    mask_src_flag = ds.g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = ds.g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)
    valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
    valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)

    return DataSplits(
        train=(src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]),
        val=(src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]),
        test=(src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]),
        test_new_new=(src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]),
        test_new_old=(src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]),
        val_time=val_time,
        test_time=test_time,
    )
