import numpy as np
import torch
import torch.nn as nn

from .encoders.walk_encoder import WalkEncoder
from .layers.merge import MergeLayer
from .position.walk_pos_encoder import WalkPositionEncoder


class NeurTWs(nn.Module):
    """Tempest-powered NeurTWs model for temporal link prediction.

    Encodes nodes via multiple temporal random walks (from Tempest), using
    GRU+ODE walk encoding with walk-native positional features and contrastive
    training.
    """

    def __init__(self, n_feat, e_feat, pos_dim, pos_enc, max_walk_len,
                 num_walks_per_node, mutual, dropout_p, walk_linear_out,
                 solver, step_size, tau, logger):
        super().__init__()

        self.feat_dim = n_feat.shape[1]
        self.e_feat_dim = e_feat.shape[1]
        self.pos_dim = pos_dim
        self.model_dim = self.feat_dim + self.e_feat_dim + pos_dim
        self.out_dim = self.feat_dim
        self.tau = tau
        self.max_walk_len = max_walk_len
        self.K = num_walks_per_node
        self.mutual = mutual

        # Frozen node feature embedding (indexed by node ID)
        self.node_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(n_feat).float(), freeze=True
        )

        # Walk-native position encoder
        self.pos_encoder = WalkPositionEncoder(
            pos_enc, pos_dim, max_walk_len, num_walks_per_node
        )

        # Walk encoder (GRU+ODE feature encoding → projection → pooling)
        self.walk_encoder = WalkEncoder(
            feat_dim=self.model_dim,   # WalkEncoder.feat_dim = concatenated model_dim
            pos_dim=pos_dim,
            model_dim=self.model_dim,
            out_dim=self.out_dim,
            logger=logger,
            mutual=mutual,
            dropout_p=dropout_p,
            walk_linear_out=walk_linear_out,
            solver=solver,
            step_size=step_size,
        )

        # Affinity scoring head
        self.affinity_score = MergeLayer(self.out_dim, self.out_dim, self.out_dim, 1)

        # Walk storage (populated by set_walks before each time window)
        self._walk_nodes = None       # (N, K, L) long
        self._walk_times = None       # (N, K, L) float
        self._walk_lens = None        # (N, K) long
        self._walk_edge_feats = None  # (N, K, L-1, E) float or None
        self._node2idx = {}

    # ------------------------------------------------------------------
    # Walk management
    # ------------------------------------------------------------------

    def set_walks(self, nodes, times, lens, edge_feats, active_node_ids):
        """Store precomputed walks and build node-ID → index mapping.

        Args:
            nodes:           (N, K, L) numpy int array — walk node IDs
            times:           (N, K, L) numpy float array — walk timestamps
            lens:            (N, K) numpy int array — actual walk lengths
            edge_feats:      (N, K, L-1, E) numpy float array, or None
            active_node_ids: (N,) numpy int array — node ID for each row,
                             from the backend's sorted active-node list.
        """
        device = next(self.parameters()).device
        self._walk_nodes = torch.from_numpy(nodes).long().to(device)
        self._walk_times = torch.from_numpy(times).float().to(device)
        self._walk_lens = torch.from_numpy(lens).long().to(device)
        if edge_feats is not None:
            self._walk_edge_feats = torch.from_numpy(edge_feats).float().to(device)
        else:
            self._walk_edge_feats = None

        self._node2idx = {int(nid): i for i, nid in enumerate(active_node_ids)}

    def _get_walks(self, node_ids):
        """Slice stored walks for a batch of node IDs.

        Nodes absent from ``_node2idx`` (e.g. unseen during walk generation)
        receive zero-filled walks with length 0, which the mask will blank out.

        Returns: (nodes, times, lens, edge_feats) tensors.
        """
        device = self._walk_nodes.device
        idx = []
        missing = []
        for i, nid in enumerate(node_ids):
            pos = self._node2idx.get(int(nid))
            if pos is not None:
                idx.append(pos)
                missing.append(False)
            else:
                idx.append(0)  # placeholder index
                missing.append(True)

        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        nodes = self._walk_nodes[idx_t]
        times = self._walk_times[idx_t]
        lens = self._walk_lens[idx_t]
        ef = self._walk_edge_feats[idx_t] if self._walk_edge_feats is not None else None

        if any(missing):
            mask = torch.tensor(missing, dtype=torch.bool, device=device)
            nodes[mask] = 0
            times[mask] = 0
            lens[mask] = 0
            if ef is not None:
                ef[mask] = 0

        return nodes, times, lens, ef

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mask(self, walk_nodes, walk_lens):
        """(B, K, L) bool — True for valid (within length, non-padding) positions."""
        B, K, L = walk_nodes.shape
        pos_grid = torch.arange(L, device=walk_nodes.device).view(1, 1, L)
        return (pos_grid < walk_lens.unsqueeze(-1)) & (walk_nodes != 0)

    def _pad_edge_features(self, edge_feats, B, K, L, device):
        """Zero-pad edge features at position 0: (B,K,L-1,E) → (B,K,L,E)."""
        if edge_feats is not None:
            pad = torch.zeros(B, K, 1, self.e_feat_dim, device=device)
            return torch.cat([pad, edge_feats], dim=2)
        return torch.zeros(B, K, L, self.e_feat_dim, device=device)

    def _encode_walks(self, walk_nodes, walk_times, walk_lens, walk_edge_feats,
                      pos_features, pool=True):
        """Run the full walk-encoding pipeline for one set of nodes.

        Returns (B, out_dim) if pool=True, else (B, K, attn_dim).
        """
        B, K, L = walk_nodes.shape
        device = walk_nodes.device
        node_feats = self.node_embedding(walk_nodes)
        edge_feats = self._pad_edge_features(walk_edge_feats, B, K, L, device)
        mask = self._build_mask(walk_nodes, walk_lens)
        return self.walk_encoder.forward_one_node(
            node_feats, edge_feats, pos_features, walk_times, mask, pool=pool,
        )

    def _compute_pair_embeddings(self, src_walks, tgt_walks):
        """Encode a (src, tgt) pair with position encoding and optional mutual attention.

        Returns: (src_embed, tgt_embed) each (B, out_dim).
        """
        src_n, src_t, src_l, src_ef = src_walks
        tgt_n, tgt_t, tgt_l, tgt_ef = tgt_walks

        src_pos, tgt_pos = self.pos_encoder(src_n, tgt_n, src_l, tgt_l)

        if self.mutual:
            src_walk_emb = self._encode_walks(src_n, src_t, src_l, src_ef, src_pos, pool=False)
            tgt_walk_emb = self._encode_walks(tgt_n, tgt_t, tgt_l, tgt_ef, tgt_pos, pool=False)
            return self.walk_encoder.mutual_query(src_walk_emb, tgt_walk_emb)

        src_emb = self._encode_walks(src_n, src_t, src_l, src_ef, src_pos)
        tgt_emb = self._encode_walks(tgt_n, tgt_t, tgt_l, tgt_ef, tgt_pos)
        return src_emb, tgt_emb

    def _encode_with_cross(self, node_walks, cross_walks):
        """Encode a node using cross-perspective position features from another node.

        Returns: (B, out_dim).
        """
        n, t, l, ef = node_walks
        cross_n, _, cross_l, _ = cross_walks
        # pos_encoder(src, tgt) → (src_pos, tgt_pos)
        # We want tgt_pos: own from node's walks, cross from cross_node's walks
        _, node_pos = self.pos_encoder(cross_n, n, cross_l, l)
        return self._encode_walks(n, t, l, ef, node_pos)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contrast(self, src_ids, dst_ids, neg_ids):
        """Compute contrastive (InfoNCE) loss.

        Args:
            src_ids: (B,) source node IDs (numpy array)
            dst_ids: (B,) positive target node IDs
            neg_ids: (B, num_negs) or (B,) negative node IDs

        Returns: scalar loss tensor.
        """
        src_walks = self._get_walks(src_ids)
        tgt_walks = self._get_walks(dst_ids)

        src_embed, tgt_embed = self._compute_pair_embeddings(src_walks, tgt_walks)

        # Positive score
        pos_logit, _ = self.affinity_score(src_embed, tgt_embed)  # (B, 1)
        pos_score = torch.exp(pos_logit / self.tau)

        # Unpack negatives into a list of (B,) arrays
        neg_ids = np.asarray(neg_ids)
        if neg_ids.ndim == 1:
            neg_list = [neg_ids]
        else:
            neg_list = [neg_ids[:, i] for i in range(neg_ids.shape[1])]

        neg_score_sum = torch.zeros_like(pos_score)
        for neg in neg_list:
            neg_walks = self._get_walks(neg)
            neg_embed = self._encode_with_cross(neg_walks, src_walks)
            neg_logit, _ = self.affinity_score(src_embed, neg_embed)
            neg_score_sum = neg_score_sum + torch.exp(neg_logit / self.tau)

        # InfoNCE: -log( exp(pos/τ) / (exp(pos/τ) + Σ exp(neg/τ)) )
        loss = -torch.log(pos_score / (pos_score + neg_score_sum + 1e-8))
        return loss.mean()

    def inference(self, src_ids, dst_ids, neg_ids, ts=None, e_idx=None):
        """Compute positive and negative probabilities for evaluation.

        Args:
            src_ids: (B,) source node IDs
            dst_ids: (B,) positive target node IDs
            neg_ids: (B,) negative node IDs (single negative)
            ts:    ignored (API compatibility with evaluator)
            e_idx: ignored (API compatibility with evaluator)

        Returns:
            pos_prob: (B,) sigmoid probabilities for positive edges
            neg_prob: (B,) sigmoid probabilities for negative edges
        """
        src_walks = self._get_walks(src_ids)
        tgt_walks = self._get_walks(dst_ids)
        neg_walks = self._get_walks(neg_ids)

        src_embed, tgt_embed = self._compute_pair_embeddings(src_walks, tgt_walks)
        neg_embed = self._encode_with_cross(neg_walks, src_walks)

        pos_logit, _ = self.affinity_score(src_embed, tgt_embed)
        neg_logit, _ = self.affinity_score(src_embed, neg_embed)

        return torch.sigmoid(pos_logit).squeeze(-1), torch.sigmoid(neg_logit).squeeze(-1)
