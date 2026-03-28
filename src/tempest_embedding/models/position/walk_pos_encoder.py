import torch
import torch.nn as nn


class WalkPositionEncoder(nn.Module):
    """Walk-native positional encoding using Tempest walk output.

    Two modes:
      - SAW (first-seen position): per-node minimum walk position via scatter_reduce.
      - LP  (walk-presence frequency): fraction of K walks that contain each node.

    Both modes compute own-perspective (from the root's walks) and cross-perspective
    (from the paired endpoint's walks) features, following the NeurTWs / CAWs
    set-based anonymisation principle.
    """

    def __init__(self, mode, pos_dim, max_walk_len, num_walks_per_node):
        super().__init__()
        self.mode = mode
        self.pos_dim = pos_dim
        self.L = max_walk_len
        self.K = num_walks_per_node
        self.sentinel = max_walk_len  # "not found" index for SAW

        if mode == 'saw':
            num_positions = max_walk_len + 1  # 0..L-1 valid, L = sentinel
            half = pos_dim // 2
            other_half = pos_dim - half
            self.own_emb = nn.Embedding(num_positions, half)
            self.cross_emb = nn.Embedding(num_positions, other_half)
        elif mode == 'lp':
            self.mlp = nn.Sequential(
                nn.Linear(2, pos_dim),
                nn.ReLU(),
                nn.Linear(pos_dim, pos_dim),
            )
        else:
            raise ValueError(f"Unknown position encoding mode: {mode}")

    def forward(self, src_walks, tgt_walks, src_lens, tgt_lens):
        """
        Args:
            src_walks: (B, K, L) long — node IDs for source walks
            tgt_walks: (B, K, L) long — node IDs for target walks
            src_lens:  (B, K) long — actual walk lengths per source walk
            tgt_lens:  (B, K) long — actual walk lengths per target walk

        Returns:
            src_pos: (B, K, L, pos_dim) — positional features for source
            tgt_pos: (B, K, L, pos_dim) — positional features for target
        """
        src_walks = src_walks.long()
        tgt_walks = tgt_walks.long()
        src_lens = src_lens.long()
        tgt_lens = tgt_lens.long()

        src_valid = self._valid_mask(src_walks, src_lens)
        tgt_valid = self._valid_mask(tgt_walks, tgt_lens)

        if self.mode == 'saw':
            return self._forward_saw(src_walks, tgt_walks, src_valid, tgt_valid)
        else:
            return self._forward_lp(src_walks, tgt_walks, src_valid, tgt_valid)

    def _valid_mask(self, walks, lens):
        """(B, K, L) bool — True where position is within length and non-padding."""
        B, K, L = walks.shape
        pos_grid = torch.arange(L, device=walks.device).view(1, 1, L)
        return (pos_grid < lens.unsqueeze(-1)) & (walks != 0)

    # ------------------------------------------------------------------
    # SAW: first-seen walk position
    # ------------------------------------------------------------------

    def _forward_saw(self, src_walks, tgt_walks, src_valid, tgt_valid):
        B, K, L = src_walks.shape
        M = K * L

        num_slots = max(src_walks.max().item(), tgt_walks.max().item()) + 1

        src_table = self._saw_table(src_walks, src_valid, num_slots)
        tgt_table = self._saw_table(tgt_walks, tgt_valid, num_slots)

        src_flat = src_walks.reshape(B, M)
        tgt_flat = tgt_walks.reshape(B, M)

        # Own perspective: earliest position in own walks
        src_own = src_table.gather(1, src_flat).view(B, K, L)
        tgt_own = tgt_table.gather(1, tgt_flat).view(B, K, L)

        # Cross perspective: earliest position in paired endpoint's walks
        src_cross = tgt_table.gather(1, src_flat).view(B, K, L)
        tgt_cross = src_table.gather(1, tgt_flat).view(B, K, L)

        # Embed each index and concatenate own + cross
        src_pos = torch.cat([self.own_emb(src_own), self.cross_emb(src_cross)], dim=-1)
        tgt_pos = torch.cat([self.own_emb(tgt_own), self.cross_emb(tgt_cross)], dim=-1)

        # Zero out invalid positions
        src_pos = src_pos * src_valid.unsqueeze(-1).float()
        tgt_pos = tgt_pos * tgt_valid.unsqueeze(-1).float()

        return src_pos, tgt_pos

    def _saw_table(self, walks, valid, num_slots):
        """Per-node minimum walk position table.

        Returns (B, num_slots) long tensor.  Unseen nodes have value self.sentinel.
        """
        B, K, L = walks.shape
        M = K * L
        device = walks.device

        flat = walks.reshape(B, M)

        # Position indices: 0..L-1 tiled K times
        pos = torch.arange(L, device=device).repeat(K).unsqueeze(0).expand(B, M)

        # Sentinel for invalid tokens
        valid_flat = valid.reshape(B, M)
        pos = torch.where(valid_flat, pos, torch.full_like(pos, self.sentinel))

        table = torch.full((B, num_slots), self.sentinel, dtype=torch.long, device=device)
        table.scatter_reduce_(1, flat, pos, reduce="amin", include_self=True)
        return table

    # ------------------------------------------------------------------
    # LP: walk-presence frequency
    # ------------------------------------------------------------------

    def _forward_lp(self, src_walks, tgt_walks, src_valid, tgt_valid):
        B, K, L = src_walks.shape
        M = K * L

        num_slots = max(src_walks.max().item(), tgt_walks.max().item()) + 1

        src_table = self._lp_table(src_walks, src_valid, num_slots)
        tgt_table = self._lp_table(tgt_walks, tgt_valid, num_slots)

        src_flat = src_walks.reshape(B, M)
        tgt_flat = tgt_walks.reshape(B, M)

        # Own + cross frequencies
        src_own = src_table.gather(1, src_flat).view(B, K, L)
        src_cross = tgt_table.gather(1, src_flat).view(B, K, L)
        tgt_own = tgt_table.gather(1, tgt_flat).view(B, K, L)
        tgt_cross = src_table.gather(1, tgt_flat).view(B, K, L)

        # Stack (own, cross) and pass through MLP
        src_feat = torch.stack([src_own, src_cross], dim=-1)  # (B, K, L, 2)
        tgt_feat = torch.stack([tgt_own, tgt_cross], dim=-1)
        src_pos = self.mlp(src_feat)  # (B, K, L, pos_dim)
        tgt_pos = self.mlp(tgt_feat)

        # Zero out invalid positions
        src_pos = src_pos * src_valid.unsqueeze(-1).float()
        tgt_pos = tgt_pos * tgt_valid.unsqueeze(-1).float()

        return src_pos, tgt_pos

    def _lp_table(self, walks, valid, num_slots):
        """Per-node walk-presence frequency table.

        For each node v, counts how many of the K walks contain v at least once,
        then divides by K.  Returns (B, num_slots) float tensor.
        """
        B, K, L = walks.shape
        M = K * L
        device = walks.device

        # --- deduplicate within each walk: mark first occurrence only ---
        BK = B * K
        walk_flat = walks.reshape(BK, L)
        valid_wf = valid.reshape(BK, L)

        sorted_nodes, sort_idx = walk_flat.sort(dim=1)
        sorted_valid = valid_wf.gather(1, sort_idx)

        # First in each run of identical node IDs (within a single walk)
        first_occ = torch.ones(BK, L, dtype=torch.bool, device=device)
        first_occ[:, 1:] = sorted_nodes[:, 1:] != sorted_nodes[:, :-1]
        first_occ = first_occ & sorted_valid

        # Map back to original token order
        unsort_idx = sort_idx.argsort(dim=1)
        first_occ_orig = first_occ.gather(1, unsort_idx)  # (BK, L)

        # --- scatter-add first-occurrence marks by node ID ---
        flat = walks.reshape(B, M)
        first_flat = first_occ_orig.reshape(B, M).float()

        table = torch.zeros(B, num_slots, device=device)
        table.scatter_add_(1, flat, first_flat)
        table = table / self.K

        return table
