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
        self.logger = logger
        self._contrast_calls = 0

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
        t0_total = perf_counter()
        timing = {'position': 0.0, 'model': 0.0}
        stage_timing = {
            'pair_encode': 0.0,
            'pos_affinity': 0.0,
            'neg_encode': 0.0,
            'neg_affinity': 0.0,
        }

        t0 = perf_counter()
        src_embed, tgt_embed, pair_timing = self._compute_pair_embeddings(src_walks, dst_walks)
        stage_timing['pair_encode'] += perf_counter() - t0
        timing['position'] += pair_timing['position']
        timing['model'] += pair_timing['model']

        t0 = perf_counter()
        pos_logit, _ = self.affinity_score(src_embed, tgt_embed)
        pos_score = torch.exp(pos_logit / self.tau)
        stage_timing['pos_affinity'] += perf_counter() - t0

        neg_score_sum = torch.zeros_like(pos_score)

        for neg_walks in neg_walks_list:
            t0 = perf_counter()
            neg_embed, neg_timing = self._encode_with_cross(neg_walks, src_walks)
            stage_timing['neg_encode'] += perf_counter() - t0
            timing['position'] += neg_timing['position']
            timing['model'] += neg_timing['model']

            t0 = perf_counter()
            neg_logit, _ = self.affinity_score(src_embed, neg_embed)
            neg_score_sum = neg_score_sum + torch.exp(neg_logit / self.tau)
            stage_timing['neg_affinity'] += perf_counter() - t0

        timing['model'] += (
            stage_timing['pos_affinity']
            + stage_timing['neg_affinity']
        )
        loss = -torch.log(pos_score / (pos_score + neg_score_sum + 1e-8))
        self._contrast_calls += 1
        total_elapsed = perf_counter() - t0_total
        if self.logger is not None and (total_elapsed > 2.0 or (self._contrast_calls % 50) == 0):
            self.logger.info(
                'NeurTWs.contrast: call=%d batch=%d negs=%d total=%.3fs position=%.3fs model=%.3fs '
                'stages(pair=%.3fs pos_aff=%.3fs neg_enc=%.3fs neg_aff=%.3fs)',
                self._contrast_calls,
                src_walks[0].shape[0],
                len(neg_walks_list),
                total_elapsed,
                timing['position'],
                timing['model'],
                stage_timing['pair_encode'],
                stage_timing['pos_affinity'],
                stage_timing['neg_encode'],
                stage_timing['neg_affinity'],
            )
        return loss.mean(), timing

    def inference(self, src_walks, dst_walks, neg_walks):
        src_embed, tgt_embed, pair_timing = self._compute_pair_embeddings(src_walks, dst_walks)
        neg_embed, neg_timing = self._encode_with_cross(neg_walks, src_walks)

        pos_logit, _ = self.affinity_score(src_embed, tgt_embed)
        neg_logit, _ = self.affinity_score(src_embed, neg_embed)

        return torch.sigmoid(pos_logit).squeeze(-1), torch.sigmoid(neg_logit).squeeze(-1)
