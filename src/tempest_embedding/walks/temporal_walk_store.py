from __future__ import annotations

import numpy as np
import torch
from time import perf_counter

from temporal_random_walk import TemporalRandomWalk


class TemporalWalkStore:
    """
    Unified walk store + Tempest backend.

    Responsibilities:
    - own the TemporalRandomWalk object
    - ingest graph edges over time
    - generate walks for current graph state
    - expose fast lookup by node ID

    Mapping correctness relies on Tempest's get_node_ids() order, which matches
    the reshaped walk rows.
    """

    def __init__(self, args, device: torch.device | None = None, logger=None):
        self.args = args
        self.device = device or torch.device('cpu')
        self.logger = logger

        enable_weight = args.walk_bias in ["ExponentialWeight", "SpatioTemporal"]
        enable_tn2v = args.walk_bias == "TemporalNode2Vec"

        self.trw = TemporalRandomWalk(
            is_directed=False,
            use_gpu=args.walk_use_gpu,
            max_time_capacity=args.max_time_capacity,
            enable_weight_computation=enable_weight,
            enable_temporal_node2vec=enable_tn2v,
            timescale_bound=args.timescale_bound,
            walk_padding_value=args.walk_padding_value,
            shuffle_walk_order=False,
        )

        self.nodes: torch.Tensor | None = None
        self.times: torch.Tensor | None = None
        self.lens: torch.Tensor | None = None
        self.edge_feats: torch.Tensor | None = None

        self.node_ids: torch.Tensor | None = None
        self.node_index: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Graph ingestion
    # ------------------------------------------------------------------

    def add_edges(self, src, dst, ts, edge_feats=None):
        t0 = perf_counter()
        if edge_feats is not None:
            edge_feats = edge_feats.astype(np.float32)

        self.trw.add_multiple_edges(
            src.astype(np.int32),
            dst.astype(np.int32),
            ts.astype(np.int64),
            edge_feats,
        )
        if self.logger is not None:
            self.logger.info(
                'TemporalWalkStore.add_edges: added=%d range_ts=[%s, %s] edge_feat=%s elapsed=%.2fs',
                len(src),
                int(ts.min()) if len(ts) else 'NA',
                int(ts.max()) if len(ts) else 'NA',
                edge_feats is not None,
                perf_counter() - t0,
            )

    # ------------------------------------------------------------------
    # Walk generation
    # ------------------------------------------------------------------

    def build(self):
        t0 = perf_counter()
        if self.logger is not None:
            self.logger.info('TemporalWalkStore.build: starting walk generation (walk_len=%d walks_per_node=%d)', self.args.max_walk_len, self.args.num_walks_per_node)
        nodes, times, lens, edge_feats = self.trw.get_random_walks_and_times_for_all_nodes(
            max_walk_len=self.args.max_walk_len,
            walk_bias=self.args.walk_bias,
            num_walks_per_node=self.args.num_walks_per_node,
            initial_edge_bias=self.args.initial_edge_bias,
            walk_direction=self.args.walk_direction,
        )

        node_ids = self.trw.get_node_ids()

        K = self.args.num_walks_per_node
        L = self.args.max_walk_len
        N = len(node_ids)

        expected_num_walks = N * K
        if nodes.shape[0] != expected_num_walks:
            raise RuntimeError(
                f'Walk count mismatch: got {nodes.shape[0]}, expected {expected_num_walks} '
                f'for {N} nodes and K={K}'
            )

        nodes = nodes.reshape(N, K, L)
        times = times.reshape(N, K, L)
        lens = lens.reshape(N, K)

        if edge_feats is not None:
            edge_feats = edge_feats.reshape(N, K, L - 1, -1)

        self.nodes = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        self.times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        self.lens = torch.as_tensor(lens, dtype=torch.long, device=self.device)
        self.edge_feats = (
            torch.as_tensor(edge_feats, dtype=torch.float32, device=self.device)
            if edge_feats is not None else None
        )

        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)

        max_id = int(self.node_ids.max().item()) if self.node_ids.numel() > 0 else -1
        self.node_index = torch.full(
            (max_id + 1,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.node_index[self.node_ids] = torch.arange(N, device=self.device)

        if self.logger is not None:
            self.logger.info(
                'TemporalWalkStore.build: done nodes=%d walks=%d tensor_nodes=%s tensor_times=%s edge_feats=%s elapsed=%.2fs',
                N,
                expected_num_walks,
                tuple(self.nodes.shape),
                tuple(self.times.shape),
                tuple(self.edge_feats.shape) if self.edge_feats is not None else None,
                perf_counter() - t0,
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, node_ids):
        if self.node_index is None:
            raise RuntimeError('TemporalWalkStore.build() must be called before get().')

        node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)

        if node_ids.numel() == 0:
            empty_nodes = self.nodes[:0]
            empty_times = self.times[:0]
            empty_lens = self.lens[:0]
            empty_edge_feats = self.edge_feats[:0] if self.edge_feats is not None else None
            return empty_nodes, empty_times, empty_lens, empty_edge_feats

        if int(node_ids.max().item()) >= self.node_index.shape[0]:
            missing = node_ids[node_ids >= self.node_index.shape[0]]
            raise KeyError(f'Nodes missing in TemporalWalkStore: {missing.tolist()}')

        idx = self.node_index[node_ids]
        if (idx < 0).any():
            missing = node_ids[idx < 0]
            raise KeyError(f'Nodes missing in TemporalWalkStore: {missing.tolist()}')

        nodes = self.nodes[idx]
        times = self.times[idx]
        lens = self.lens[idx]
        edge_feats = self.edge_feats[idx] if self.edge_feats is not None else None
        return nodes, times, lens, edge_feats

    def has(self, node_ids) -> torch.Tensor:
        if self.node_index is None:
            raise RuntimeError('TemporalWalkStore.build() must be called before has().')

        node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)
        in_bounds = node_ids < self.node_index.shape[0]
        out = torch.zeros_like(node_ids, dtype=torch.bool, device=self.device)
        out[in_bounds] = self.node_index[node_ids[in_bounds]] >= 0
        return out

    def valid_nodes(self) -> torch.Tensor:
        if self.node_ids is None:
            raise RuntimeError('TemporalWalkStore.build() must be called before valid_nodes().')
        return self.node_ids

    def get_num_nodes(self) -> int:
        return 0 if self.node_ids is None else int(self.node_ids.numel())

    def get_num_edges(self):
        return self.trw.get_edge_count()

    def __len__(self):
        return self.get_num_nodes()
