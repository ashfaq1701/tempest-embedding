import numpy as np


class WalkBatcher:
    def __init__(self, num_walks_per_node, max_walk_len):
        self.num_walks_per_node = num_walks_per_node
        self.max_walk_len = max_walk_len

    def reshape_walks(self, nodes, times, lens, edge_feats=None):
        """
        Convert Tempest output:
            (num_nodes * K, L)

        → structured form:
            (num_nodes, K, L)
        """

        num_walks = nodes.shape[0]
        K = self.num_walks_per_node

        assert num_walks % K == 0, "Walk count must be divisible by walks_per_node"

        num_nodes = num_walks // K

        nodes = nodes.reshape(num_nodes, K, self.max_walk_len)
        times = times.reshape(num_nodes, K, self.max_walk_len)
        lens = lens.reshape(num_nodes, K)

        if edge_feats is not None:
            edge_feats = edge_feats.reshape(num_nodes, K, self.max_walk_len - 1, -1)

        return nodes, times, lens, edge_feats

    def sample_node_batch(self, nodes, times, lens, edge_feats, batch_size):
        """
        Sample a batch of nodes.

        Input:
            nodes: (N, K, L)

        Output:
            batch_nodes: (B, K, L)
        """

        num_nodes = nodes.shape[0]

        if batch_size >= num_nodes:
            return nodes, times, lens, edge_feats

        idx = np.random.choice(num_nodes, size=batch_size, replace=False)

        batch_nodes = nodes[idx]
        batch_times = times[idx]
        batch_lens = lens[idx]

        if edge_feats is not None:
            batch_edge_feats = edge_feats[idx]
        else:
            batch_edge_feats = None

        return batch_nodes, batch_times, batch_lens, batch_edge_feats