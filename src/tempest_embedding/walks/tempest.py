import numpy as np

from temporal_random_walk import TemporalRandomWalk


class TempestWalkBackend:
    def __init__(self, args):
        self.args = args

        # ---- Bias-dependent flags ----
        enable_weight = args.walk_bias in ["ExponentialWeight", "SpatioTemporal"]
        enable_tn2v = args.walk_bias == "TemporalNode2Vec"

        self.tw = TemporalRandomWalk(
            is_directed=False,
            use_gpu=args.walk_use_gpu,
            max_time_capacity=args.max_time_capacity,
            enable_weight_computation=enable_weight,
            enable_temporal_node2vec=enable_tn2v,
            timescale_bound=args.timescale_bound,
            walk_padding_value=args.walk_padding_value,
        )

    # -----------------------------
    # Ingestion
    # -----------------------------
    def add_edges(self, sources, targets, timestamps, edge_features=None):
        """
        sources: np.ndarray[int32]
        targets: np.ndarray[int32]
        timestamps: np.ndarray[int64]
        edge_features: np.ndarray[float32] shape [num_edges, feat_dim] or None
        """

        if edge_features is not None:
            edge_features = edge_features.astype(np.float32)

        self.tw.add_multiple_edges(
            sources.astype(np.int32),
            targets.astype(np.int32),
            timestamps.astype(np.int64),
            edge_features
        )

    # -----------------------------
    # Walk generation
    # -----------------------------
    def generate_walks(self):
        nodes, times, lens, edge_feats = self.tw.get_random_walks_and_times_for_all_nodes(
            max_walk_len=self.args.max_walk_len,
            walk_bias=self.args.walk_bias,
            num_walks_per_node=self.args.num_walks_per_node,
            initial_edge_bias=self.args.initial_edge_bias,
            walk_direction=self.args.walk_direction,
        )

        return nodes, times, lens, edge_feats

    # -----------------------------
    # Utility
    # -----------------------------
    def get_num_nodes(self):
        return self.tw.get_node_count()

    def get_num_edges(self):
        return self.tw.get_edge_count()
