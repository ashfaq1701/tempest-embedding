import numpy as np
import torch
import torch.nn as nn

from .encoders.walk_encoder import WalkEncoder
from .layers.merge import MergeLayer
from .position.walk_pos_encoder import WalkPositionEncoder
from ..utils.misc import PAD_NODE_ID


class NeurTWs(nn.Module):
    """Tempest-powered NeurTWs model for temporal link prediction."""

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

        self.node_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(n_feat).float(), freeze=True
        )

        self.pos_encoder = WalkPositionEncoder(
            pos_enc, pos_dim, max_walk_len, num_walks_per_node
        )

        self.walk_encoder = WalkEncoder(
            feat_dim=self.model_dim,
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

        self.affinity_score = MergeLayer(self.out_dim, self.out_dim, self.out_dim, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mask(self, walk_nodes, walk_lens):
        B, K, L = walk_nodes.shape
        pos_grid = torch.arange(L, device=walk_nodes.device).view(1, 1, L)
        return (pos_grid < walk_lens.unsqueeze(-1)) & (walk_nodes != PAD_NODE_ID)

    def _pad_edge_features(self, edge_feats, B, K, L, device):
        if edge_feats is not None:
            pad = torch.zeros(B, K, 1, self.e_feat_dim, device=device)
            return torch.cat([pad, edge_feats], dim=2)
        return torch.zeros(B, K, L, self.e_feat_dim, device=device)

    def _encode_walks(self, walk_nodes, walk_times, walk_lens, walk_edge_feats,
                      pos_features, pool=True):
        B, K, L = walk_nodes.shape
        device = walk_nodes.device

        safe_nodes = torch.where(
            walk_nodes == PAD_NODE_ID,
            torch.zeros_like(walk_nodes),
            walk_nodes,
        )

        node_feats = self.node_embedding(safe_nodes)
        edge_feats = self._pad_edge_features(walk_edge_feats, B, K, L, device)
        mask = self._build_mask(walk_nodes, walk_lens)

        return self.walk_encoder.forward_one_node(
            node_feats, edge_feats, pos_features, walk_times, mask, pool=pool,
        )

    def _compute_pair_embeddings(self, src_walks, tgt_walks):
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
        n, t, l, ef = node_walks
        cross_n, _, cross_l, _ = cross_walks
        _, node_pos = self.pos_encoder(cross_n, n, cross_l, l)
        return self._encode_walks(n, t, l, ef, node_pos)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contrast(self, src_walks, dst_walks, neg_walks_list):
        src_embed, tgt_embed = self._compute_pair_embeddings(src_walks, dst_walks)

        pos_logit, _ = self.affinity_score(src_embed, tgt_embed)
        pos_score = torch.exp(pos_logit / self.tau)

        neg_score_sum = torch.zeros_like(pos_score)
        for neg_walks in neg_walks_list:
            neg_embed = self._encode_with_cross(neg_walks, src_walks)
            neg_logit, _ = self.affinity_score(src_embed, neg_embed)
            neg_score_sum = neg_score_sum + torch.exp(neg_logit / self.tau)

        loss = -torch.log(pos_score / (pos_score + neg_score_sum + 1e-8))
        return loss.mean()

    def inference(self, src_walks, dst_walks, neg_walks):
        src_embed, tgt_embed = self._compute_pair_embeddings(src_walks, dst_walks)
        neg_embed = self._encode_with_cross(neg_walks, src_walks)

        pos_logit, _ = self.affinity_score(src_embed, tgt_embed)
        neg_logit, _ = self.affinity_score(src_embed, neg_embed)

        return torch.sigmoid(pos_logit).squeeze(-1), torch.sigmoid(neg_logit).squeeze(-1)
