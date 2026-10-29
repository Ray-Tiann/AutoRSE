from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_intsr_masks(event_valid: torch.Tensor,
                      event_times: torch.Tensor,
                      event_item_ids: torch.Tensor,
                      event_is_search: torch.Tensor,
                      current_times: torch.Tensor,
                      current_item_ids: torch.Tensor = None,
                      event_group_ids: torch.Tensor = None,
                      current_group_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """Build the causal, session-wise, and invalid-Q masks used by IntSR.

    Each historical action contributes three ordinary tokens (S, I, F) and
    one query token.  A final query token is appended for the current request.
    Boolean mask values follow PyTorch's natural convention here: ``True``
    means that an attention edge is visible.
    """
    if event_valid.ndim != 2:
        raise ValueError("event_valid must have shape [batch, events]")

    batch_size, num_events = event_valid.shape
    device = event_valid.device

    if event_times.shape != event_valid.shape:
        raise ValueError("event_times must match event_valid")
    if event_item_ids.shape != event_valid.shape:
        raise ValueError("event_item_ids must match event_valid")
    if event_is_search.shape != event_valid.shape:
        raise ValueError("event_is_search must match event_valid")
    if current_times.shape != (batch_size,):
        raise ValueError("current_times must have shape [batch]")
    if current_item_ids is not None:
        if current_item_ids.ndim == 1:
            current_item_ids = current_item_ids.unsqueeze(1)
        if current_item_ids.ndim != 2 or current_item_ids.size(0) != batch_size:
            raise ValueError(
                "current_item_ids must have shape [batch] or "
                "[batch, candidates]")
    if (event_group_ids is None) != (current_group_ids is None):
        raise ValueError(
            "event_group_ids and current_group_ids must be provided together")
    if event_group_ids is not None:
        if event_group_ids.shape != event_valid.shape:
            raise ValueError("event_group_ids must match event_valid")
        if current_group_ids.shape != (batch_size,):
            raise ValueError("current_group_ids must have shape [batch]")
        event_group_ids = event_group_ids.to(device=device, dtype=torch.long)
        current_group_ids = current_group_ids.to(device=device, dtype=torch.long)

    action_index = torch.arange(num_events, device=device)

    # S -> Q -> I -> F positions are shared even though QDB stores ordinary
    # tokens and query tokens in separate tensors. The current request adds
    # S_{n+1} -> Q_{n+1}, with no observed target item or feedback yet.
    history_position = action_index * 4
    ordinary_position = torch.stack([
        history_position,
        history_position + 2,
        history_position + 3,
    ], dim=1).reshape(-1)
    ordinary_position = torch.cat([
        ordinary_position,
        torch.tensor([num_events * 4], device=device),
    ])
    query_position = torch.arange(num_events + 1, device=device) * 4 + 1

    ordinary_valid = torch.cat([
        event_valid.repeat_interleave(3, dim=1),
        torch.ones((batch_size, 1), dtype=torch.bool, device=device),
    ], dim=1)
    query_row_valid = torch.cat([
        event_valid,
        torch.ones((batch_size, 1), dtype=torch.bool, device=device),
    ], dim=1)

    # Fall back to matching time/item only for callers without reconstructed
    # group IDs. The identity diagonal remains visible within an action.
    same_session = (
        event_valid[:, :, None]
        & event_valid[:, None, :]
        & (event_times[:, :, None] == event_times[:, None, :])
        & (event_item_ids[:, :, None] == event_item_ids[:, None, :])
    )
    if event_group_ids is not None:
        valid_groups = event_group_ids.ne(0)
        same_session = (
            event_valid[:, :, None]
            & event_valid[:, None, :]
            & valid_groups[:, :, None]
            & valid_groups[:, None, :]
            & (event_group_ids[:, :, None] == event_group_ids[:, None, :])
        )
    different_action = (
        action_index[:, None] != action_index[None, :]
    ).unsqueeze(0)
    blocked_session = same_session & different_action
    # With group IDs, hide only the current request's group; timestamp-only
    # fallback is retained for unit-test and legacy callers.
    current_session_history = (
        event_valid
        & (event_times == current_times[:, None])
    )
    if event_group_ids is not None:
        grouped_history = event_valid & event_group_ids.eq(
            current_group_ids[:, None])
        current_session_history = torch.where(
            current_group_ids.ne(0)[:, None],
            grouped_history,
            current_session_history,
        )

    ordinary_session_block = torch.zeros(
        (batch_size, num_events * 3 + 1, num_events * 3 + 1),
        dtype=torch.bool,
        device=device,
    )
    ordinary_session_block[:, :num_events * 3, :num_events * 3] = (
        blocked_session.repeat_interleave(3, dim=1).repeat_interleave(3, dim=2)
    )
    ordinary_session_block[:, -1, :num_events * 3] = (
        current_session_history.repeat_interleave(3, dim=1)
    )
    x1_mask = (
        ordinary_position[:, None] >= ordinary_position[None, :]
    ).unsqueeze(0)
    x1_mask = x1_mask & ordinary_valid[:, :, None]
    x1_mask = x1_mask & ordinary_valid[:, None, :]
    x1_mask = x1_mask & ~ordinary_session_block

    q_to_x1_causal = (
        query_position[:, None] >= ordinary_position[None, :]
    ).unsqueeze(0)
    q_to_x1_session = torch.zeros(
        (batch_size, num_events + 1, num_events * 3 + 1),
        dtype=torch.bool,
        device=device,
    )
    q_to_x1_session[:, :num_events, :num_events * 3] = (
        blocked_session.repeat_interleave(3, dim=2)
    )
    q_to_x1_session[:, -1, :num_events * 3] = (
        current_session_history.repeat_interleave(3, dim=1)
    )
    q_to_x1_mask = q_to_x1_causal
    q_to_x1_mask = q_to_x1_mask & query_row_valid[:, :, None]
    q_to_x1_mask = q_to_x1_mask & ordinary_valid[:, None, :]
    q_to_x1_mask = q_to_x1_mask & ~q_to_x1_session

    q_to_q_causal = (
        query_position[:, None] >= query_position[None, :]
    ).unsqueeze(0)
    q_to_q_session = torch.zeros(
        (batch_size, num_events + 1, num_events + 1),
        dtype=torch.bool,
        device=device,
    )
    q_to_q_session[:, :num_events, :num_events] = blocked_session
    q_to_q_session[:, -1, :num_events] = current_session_history

    # Only natural-language search or augmented queries may be historical
    # keys. Invalid/universal tokens retain their self edge so that their own
    # representation can still be computed.
    query_key_valid = torch.cat([
        event_valid & event_is_search,
        torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
    ], dim=1)
    identity = torch.eye(num_events + 1,
                         dtype=torch.bool,
                         device=device).unsqueeze(0)
    q_key_visibility = query_key_valid[:, None, :] | identity

    q_to_q_mask = q_to_q_causal
    q_to_q_mask = q_to_q_mask & query_row_valid[:, :, None]
    q_to_q_mask = q_to_q_mask & q_key_visibility
    q_to_q_mask = q_to_q_mask & ~q_to_q_session

    candidate_to_x1 = None
    candidate_to_q = None
    if current_item_ids is not None:
        num_candidates = current_item_ids.size(1)
        # Appendix C.2 forms the bottom-left candidate block by replicating
        # the corresponding query-placeholder attention row. Do not make
        # history visibility candidate-item-dependent.
        candidate_to_x1 = q_to_x1_mask[:, -1:, :].expand(
            -1, num_candidates, -1)
        candidate_to_q = q_to_q_mask[:, -1:, :-1].expand(
            -1, num_candidates, -1)

    return {
        "x1": x1_mask,
        "q_to_x1": q_to_x1_mask,
        "q_to_q": q_to_q_mask,
        "ordinary_valid": ordinary_valid,
        "query_valid": query_row_valid,
        "ordinary_position": ordinary_position,
        "query_position": query_position,
        "candidate_to_x1": candidate_to_x1,
        "candidate_to_q": candidate_to_q,
    }


class RelativePositionTimeBias(nn.Module):
    def __init__(self, max_position: int, time_buckets: int):
        super().__init__()
        self.max_position = max_position
        self.time_buckets = time_buckets
        self.position_bias = nn.Embedding(2 * max_position + 1, 1)
        self.time_bias = nn.Embedding(time_buckets, 1)
        nn.init.zeros_(self.position_bias.weight)
        nn.init.zeros_(self.time_bias.weight)

    def forward(self, query_position: torch.Tensor,
                key_position: torch.Tensor,
                query_times: torch.Tensor,
                key_times: torch.Tensor) -> torch.Tensor:
        relative_position = (
            query_position[:, None] - key_position[None, :]
        ).clamp(-self.max_position, self.max_position)
        relative_position = relative_position + self.max_position
        position_bias = self.position_bias(relative_position).squeeze(-1)

        delta_seconds = torch.abs(
            query_times[:, :, None] - key_times[:, None, :]
        ).to(torch.float32)
        delta_days = delta_seconds / 86400.0
        time_bucket = torch.floor(torch.log2(delta_days + 1.0)).long()
        time_bucket = time_bucket.clamp(0, self.time_buckets - 1)
        temporal_bias = self.time_bias(time_bucket).squeeze(-1)
        return temporal_bias + position_bias.unsqueeze(0)


class QueryDrivenBlock(nn.Module):
    """HSTU-style query-driven block corresponding to IntSR Eqs. (5)-(9)."""

    def __init__(self, hidden_dim: int, dropout: float,
                 max_position: int, time_buckets: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x1_projection = nn.Linear(hidden_dim, hidden_dim * 4)
        self.x2_projection = nn.Linear(hidden_dim, hidden_dim * 4)
        self.x1_output = nn.Linear(hidden_dim, hidden_dim)
        self.x2_output = nn.Linear(hidden_dim, hidden_dim)
        self.x1_aggregate_norm = nn.LayerNorm(hidden_dim)
        self.x2_aggregate_norm = nn.LayerNorm(hidden_dim)
        self.relative_bias = RelativePositionTimeBias(
            max_position=max_position,
            time_buckets=time_buckets,
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _split(projected: torch.Tensor):
        return F.silu(projected).chunk(4, dim=-1)

    def _attention(self, query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor, mask: torch.Tensor,
                   bias: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1))
        weights = F.silu(scores + bias) * mask.to(scores.dtype)
        return torch.matmul(weights, value)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                masks: Dict[str, torch.Tensor],
                x1_times: torch.Tensor, x2_times: torch.Tensor):
        w1, q1, k1, v1 = self._split(self.x1_projection(x1))
        w2, q2, k2, v2 = self._split(self.x2_projection(x2))

        x1_position = masks["ordinary_position"]
        x2_position = masks["query_position"]
        bias_11 = self.relative_bias(
            x1_position, x1_position, x1_times, x1_times)
        bias_21 = self.relative_bias(
            x2_position, x1_position, x2_times, x1_times)
        bias_22 = self.relative_bias(
            x2_position, x2_position, x2_times, x2_times)

        aggregate_1 = self._attention(
            q1, k1, v1, masks["x1"], bias_11)
        aggregate_21 = self._attention(
            q2, k1, v1, masks["q_to_x1"], bias_21)
        aggregate_22 = self._attention(
            q2, k2, v2, masks["q_to_q"], bias_22)

        y1 = self.x1_output(self.x1_aggregate_norm(aggregate_1) * w1)
        y2 = self.x2_output(
            self.x2_aggregate_norm(aggregate_21 + aggregate_22) * w2)
        # QDB keeps HSTU's external residual stream; Norm is applied to the
        # aggregated attention output in Eqs. (8)-(9), not the residual sum.
        x1 = x1 + self.dropout(y1)
        x2 = x2 + self.dropout(y2)

        x1 = x1 * masks["ordinary_valid"].unsqueeze(-1).to(x1.dtype)
        x2 = x2 * masks["query_valid"].unsqueeze(-1).to(x2.dtype)
        return x1, x2

    def forward_candidates(self, x1: torch.Tensor,
                           history_x2: torch.Tensor,
                           candidate_x2: torch.Tensor,
                           masks: Dict[str, torch.Tensor],
                           x1_times: torch.Tensor,
                           history_x2_times: torch.Tensor,
                           current_x2_times: torch.Tensor):
        """Update shared history once and independently expand current Qs.

        candidate_x2 has shape [batch, candidates, hidden]. Causality
        guarantees that historical tokens never depend on a current candidate,
        so their projections and attention are shared. Each current Q attends
        to the shared history and to itself, but never to another candidate.
        This is algebraically equivalent to running one full QDB sequence per
        candidate while retaining the KV-cache computation described by IntSR.
        """
        w1, q1, k1, v1 = self._split(self.x1_projection(x1))
        wh, qh, kh, vh = self._split(self.x2_projection(history_x2))
        wc, qc, kc, vc = self._split(self.x2_projection(candidate_x2))

        x1_position = masks["ordinary_position"]
        history_position = masks["query_position"][:-1]
        current_position = masks["query_position"][-1:]
        bias_11 = self.relative_bias(
            x1_position, x1_position, x1_times, x1_times)
        bias_h1 = self.relative_bias(
            history_position, x1_position, history_x2_times, x1_times)
        bias_hh = self.relative_bias(
            history_position, history_position,
            history_x2_times, history_x2_times)
        bias_c1 = self.relative_bias(
            current_position, x1_position, current_x2_times, x1_times)
        bias_ch = self.relative_bias(
            current_position, history_position,
            current_x2_times, history_x2_times)
        bias_cc = self.relative_bias(
            current_position, current_position,
            current_x2_times, current_x2_times)

        aggregate_1 = self._attention(
            q1, k1, v1, masks["x1"], bias_11)
        aggregate_h1 = self._attention(
            qh, k1, v1, masks["q_to_x1"][:, :-1], bias_h1)
        aggregate_hh = self._attention(
            qh, kh, vh, masks["q_to_q"][:, :-1, :-1], bias_hh)

        current_to_x1 = masks.get("candidate_to_x1")
        if current_to_x1 is None:
            current_to_x1 = masks["q_to_x1"][:, -1:, :]
        current_to_history = masks.get("candidate_to_q")
        if current_to_history is None:
            current_to_history = masks["q_to_q"][:, -1:, :-1]
        current_to_self = masks["q_to_q"][:, -1:, -1:]
        aggregate_c1 = self._attention(
            qc, k1, v1, current_to_x1, bias_c1)
        aggregate_ch = self._attention(
            qc, kh, vh, current_to_history, bias_ch)
        self_scores = (qc * kc).sum(dim=-1, keepdim=True)
        self_weights = F.silu(self_scores + bias_cc)
        self_weights = self_weights * current_to_self.to(self_scores.dtype)
        aggregate_cc = self_weights * vc

        y1 = self.x1_output(self.x1_aggregate_norm(aggregate_1) * w1)
        yh = self.x2_output(
            self.x2_aggregate_norm(aggregate_h1 + aggregate_hh) * wh)
        yc = self.x2_output(
            self.x2_aggregate_norm(
                aggregate_c1 + aggregate_ch + aggregate_cc) * wc)
        x1 = x1 + self.dropout(y1)
        history_x2 = history_x2 + self.dropout(yh)
        candidate_x2 = candidate_x2 + self.dropout(yc)

        x1 = x1 * masks["ordinary_valid"].unsqueeze(-1).to(x1.dtype)
        history_x2 = history_x2 * (
            masks["query_valid"][:, :-1].unsqueeze(-1).to(history_x2.dtype)
        )
        candidate_x2 = candidate_x2 * (
            masks["query_valid"][:, -1:].unsqueeze(-1).to(candidate_x2.dtype)
        )
        return x1, history_x2, candidate_x2



class QueryDrivenEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float,
                 max_position: int, time_buckets: int):
        super().__init__()
        self.layers = nn.ModuleList([
            QueryDrivenBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                max_position=max_position,
                time_buckets=time_buckets,
            ) for _ in range(num_layers)
        ])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                masks: Dict[str, torch.Tensor],
                x1_times: torch.Tensor, x2_times: torch.Tensor):
        for layer in self.layers:
            x1, x2 = layer(x1, x2, masks, x1_times, x2_times)
        return x1, x2

    def forward_candidates(self, x1: torch.Tensor,
                           history_x2: torch.Tensor,
                           candidate_x2: torch.Tensor,
                           masks: Dict[str, torch.Tensor],
                           x1_times: torch.Tensor,
                           history_x2_times: torch.Tensor,
                           current_x2_times: torch.Tensor):
        for layer in self.layers:
            x1, history_x2, candidate_x2 = layer.forward_candidates(
                x1, history_x2, candidate_x2, masks,
                x1_times, history_x2_times, current_x2_times)
        return x1, history_x2, candidate_x2



class DSFNet(nn.Module):
    """Compact dynamic scenario factorization network used after QDB."""

    def __init__(self, hidden_dim: int, scenario_dim: int,
                 num_scenarios: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_scenarios = num_scenarios
        self.num_layers = num_layers
        self.scenario_gate = nn.Linear(
            scenario_dim, num_scenarios * num_layers)
        self.feature_filter = nn.Sequential(
            nn.Linear(hidden_dim + scenario_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(num_scenarios, hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(num_scenarios, hidden_dim))
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        for weight in self.weights:
            nn.init.xavier_normal_(weight)

    def forward(self, features: torch.Tensor,
                scenario_features: torch.Tensor) -> torch.Tensor:
        scenario_weights = 2.0 * torch.sigmoid(
            self.scenario_gate(scenario_features))
        scenario_weights = scenario_weights.reshape(
            -1, self.num_layers, self.num_scenarios)
        filtered = features * self.feature_filter(
            torch.cat([features, scenario_features], dim=-1))

        output = filtered
        for layer_index in range(self.num_layers):
            gamma = scenario_weights[:, layer_index]
            dynamic_weight = torch.einsum(
                "bg,gij->bij", gamma, self.weights[layer_index])
            dynamic_bias = torch.einsum(
                "bg,gj->bj", gamma, self.biases[layer_index])
            transformed = torch.bmm(
                dynamic_weight, output.unsqueeze(-1)).squeeze(-1)
            transformed = F.silu(transformed + dynamic_bias)
            output = self.norms[layer_index](
                output + self.dropout(transformed))
        return output
