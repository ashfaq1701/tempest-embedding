import torch
import torch.nn as nn

from ..layers.pooling import SetPooler
from ..layers.transformer import TransformerDecoderLayer
from .feature_encoder import FeatureEncoder


class WalkEncoder(nn.Module):
    def __init__(self, feat_dim, pos_dim, model_dim, out_dim, logger, mutual=False, dropout_p=0.1,
                 walk_linear_out=False, solver='rk4', step_size=0.125):
        super().__init__()
        self.solver = solver
        self.step_size = step_size
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim
        self.attn_dim = self.model_dim // 2
        self.n_head = 8
        self.out_dim = out_dim
        self.mutual = mutual
        self.dropout_p = dropout_p
        self.logger = logger

        self.feature_encoder = FeatureEncoder(self.feat_dim, self.model_dim, self.dropout_p, self.solver, self.step_size)
        self.position_encoder = FeatureEncoder(self.pos_dim, self.pos_dim, self.dropout_p, self.solver, self.step_size)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_encoder.hidden_dim + self.position_encoder.hidden_dim, self.attn_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )
        if self.mutual:
            self.mutual_attention_src2tgt = TransformerDecoderLayer(
                d_model=self.attn_dim, nhead=self.n_head, dim_feedforward=4 * self.model_dim,
                dropout=self.dropout_p, activation='relu'
            )
            self.mutual_attention_tgt2src = TransformerDecoderLayer(
                d_model=self.attn_dim, nhead=self.n_head, dim_feedforward=4 * self.model_dim,
                dropout=self.dropout_p, activation='relu'
            )
        self.pooler = SetPooler(self.attn_dim, self.out_dim, dropout_p=self.dropout_p, walk_linear_out=walk_linear_out)

    def forward_one_node(self, hidden_embeddings, edge_features, position_features, t_records, masks=None):
        combined_features = self.aggregate(hidden_embeddings, edge_features, position_features)
        combined_features = self.feature_encoder.integrate(t_records, combined_features, masks)
        if self.pos_dim > 0:
            position_features = self.position_encoder.integrate(t_records, position_features, masks)
            combined_features = torch.cat([combined_features, position_features], dim=-1)
        x = self.projector(combined_features)
        x = self.pooler(x, agg='mean')
        return x

    def mutual_query(self, src_embed, tgt_embed):
        src_emb = self.mutual_attention_src2tgt(src_embed, tgt_embed)
        tgt_emb = self.mutual_attention_tgt2src(tgt_embed, src_embed)
        src_emb = self.pooler(src_emb)
        tgt_emb = self.pooler(tgt_emb)
        return src_emb, tgt_emb

    def aggregate(self, hidden_embeddings, edge_features, position_features):
        if position_features is None:
            assert self.pos_dim == 0
            combined_features = torch.cat([hidden_embeddings, edge_features], dim=-1)
        else:
            combined_features = torch.cat([hidden_embeddings, edge_features, position_features], dim=-1)
        assert combined_features.size(-1) == self.feat_dim
        return combined_features.to(hidden_embeddings.device)
