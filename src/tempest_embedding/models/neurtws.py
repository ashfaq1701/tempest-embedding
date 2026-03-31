from time import perf_counter

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
        timing = {'position': 0.0, 'model': 0.0}
        src_n, src_t, src_l, src_ef = src_walks
        tgt_n, tgt_t, tgt_l, tgt_ef = tgt_walks

        t0 = perf_counter()
        src_pos, tgt_pos = self.pos_encoder(src_n, tgt_n, src_l, tgt_l)
        timing['position'] += perf_counter() - t0

        if self.mutual:
            t0 = perf_counter()
            src_walk_emb = self._encode_walks(src_n, src_t, src_l, src_ef, src_pos, pool=False)
            tgt_walk_emb = self._encode_walks(tgt_n, tgt_t, tgt_l, tgt_ef, tgt_pos, pool=False)
            src_emb, tgt_emb = self.walk_encoder.mutual_query(src_walk_emb, tgt_walk_emb)
            timing['model'] += perf_counter() - t0
            return src_emb, tgt_emb, timing

        t0 = perf_counter()
        src_emb = self._encode_walks(src_n, src_t, src_l, src_ef, src_pos)
        tgt_emb = self._encode_walks(tgt_n, tgt_t, tgt_l, tgt_ef, tgt_pos)
        timing['model'] += perf_counter() - t0
        return src_emb, tgt_emb, timing

    def _encode_with_cross(self, node_walks, cross_walks):
        timing = {'position': 0.0, 'model': 0.0}
        n, t, l, ef = node_walks
        cross_n, _, cross_l, _ = cross_walks

        t0 = perf_counter()
        _, node_pos = self.pos_encoder(cross_n, n, cross_l, l)
        timing['position'] += perf_counter() - t0

        t0 = perf_counter()
        emb = self._encode_walks(n, t, l, ef, node_pos)
        timing['model'] += perf_counter() - t0
        return emb, timing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contrast(self, src_walks, dst_walks, neg_walks_list):
        timing = {'position': 0.0, 'model': 0.0}

        src_embed, tgt_embed, pair_timing = self._compute_pair_embeddings(src_walks, dst_walks)
        timing['position'] += pair_timing['position']
        timing['model'] += pair_timing['model']

        t0 = perf_counter()
        pos_logit, _ = self.affinity_score(src_embed, tgt_embed)
        pos_score = torch.exp(pos_logit / self.tau)
        timing['model'] += perf_counter() - t0
        if len(neg_walks_list) == 0:
            neg_score_sum = torch.zeros_like(pos_score)
        else:
            t0 = perf_counter()
            neg_walks = self._merge_negative_walks(neg_walks_list)
            src_repeated = self._repeat_walks(src_walks, repeats=len(neg_walks_list))
            timing['model'] += perf_counter() - t0

            neg_embed, neg_timing = self._encode_with_cross(neg_walks, src_repeated)
            timing['position'] += neg_timing['position']
            timing['model'] += neg_timing['model']

            bsz = src_embed.shape[0]
            n_negs = len(neg_walks_list)
            out_dim = src_embed.shape[-1]

            t0 = perf_counter()
            src_flat = (
                src_embed.unsqueeze(1)
                .expand(bsz, n_negs, out_dim)
                .reshape(bsz * n_negs, out_dim)
            )
            neg_logit, _ = self.affinity_score(src_flat, neg_embed)
            neg_score = torch.exp(neg_logit / self.tau).reshape(bsz, n_negs, 1)
            neg_score_sum = neg_score.sum(dim=1)
            timing['model'] += perf_counter() - t0

        loss = -torch.log(pos_score / (pos_score + neg_score_sum + 1e-8))
        return loss.mean(), timing

    def _merge_negative_walks(self, neg_walks_list):
        neg_nodes = torch.stack([walks[0] for walks in neg_walks_list], dim=1)
        neg_times = torch.stack([walks[1] for walks in neg_walks_list], dim=1)
        neg_lens = torch.stack([walks[2] for walks in neg_walks_list], dim=1)

        if neg_walks_list[0][3] is None:
            neg_edge_feats = None
        else:
            neg_edge_feats = torch.stack([walks[3] for walks in neg_walks_list], dim=1)

        bsz, n_negs, k_walks, walk_len = neg_nodes.shape
        neg_nodes = neg_nodes.reshape(bsz * n_negs, k_walks, walk_len)
        neg_times = neg_times.reshape(bsz * n_negs, k_walks, walk_len)
        neg_lens = neg_lens.reshape(bsz * n_negs, k_walks)

        if neg_edge_feats is not None:
            neg_edge_feats = neg_edge_feats.reshape(
                bsz * n_negs,
                k_walks,
                walk_len - 1,
                neg_edge_feats.shape[-1],
            )
        return neg_nodes, neg_times, neg_lens, neg_edge_feats

    def _repeat_walks(self, walks, repeats):
        nodes, times, lens, edge_feats = walks
        bsz, k_walks, walk_len = nodes.shape

        nodes = nodes.unsqueeze(1).expand(bsz, repeats, k_walks, walk_len)
        times = times.unsqueeze(1).expand(bsz, repeats, k_walks, walk_len)
        lens = lens.unsqueeze(1).expand(bsz, repeats, k_walks)

        nodes = nodes.reshape(bsz * repeats, k_walks, walk_len)
        times = times.reshape(bsz * repeats, k_walks, walk_len)
        lens = lens.reshape(bsz * repeats, k_walks)

        if edge_feats is None:
            repeated_edge_feats = None
        else:
            repeated_edge_feats = edge_feats.unsqueeze(1).expand(
                bsz,
                repeats,
                k_walks,
                walk_len - 1,
                edge_feats.shape[-1],
            )
            repeated_edge_feats = repeated_edge_feats.reshape(
                bsz * repeats,
                k_walks,
                walk_len - 1,
                edge_feats.shape[-1],
            )

        return nodes, times, lens, repeated_edge_feats

    def inference(self, src_walks, dst_walks, neg_walks):
        src_embed, tgt_embed, pair_timing = self._compute_pair_embeddings(src_walks, dst_walks)
        neg_embed, neg_timing = self._encode_with_cross(neg_walks, src_walks)

        pos_logit, _ = self.affinity_score(src_embed, tgt_embed)
        neg_logit, _ = self.affinity_score(src_embed, neg_embed)

        return torch.sigmoid(pos_logit).squeeze(-1), torch.sigmoid(neg_logit).squeeze(-1)
