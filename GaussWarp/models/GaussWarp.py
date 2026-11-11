from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor

def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_variables = int(configs.enc_in)
        self.dimension = int(getattr(configs, 'gausswarp_dimension', configs.d_model))
        self.num_heads = int(getattr(configs, 'gausswarp_num_heads', 4))
        self.num_slots = int(getattr(configs, 'gausswarp_num_slots', 12))
        self.num_layers = int(getattr(configs, 'gausswarp_num_layers', 1))
        self.num_lag_scales = int(getattr(configs, 'gausswarp_num_lag_scales', 6))
        if self.num_lag_scales < 1:
            raise ValueError('gausswarp_num_lag_scales must be positive')
        self.num_statistics = 5
        self.residual_limit = float(getattr(configs, 'gausswarp_residual_limit', 4.0))
        self.slope_limit = float(getattr(configs, 'gausswarp_slope_limit', 40.0))
        self.auxiliary_weight = float(getattr(configs, 'gausswarp_auxiliary_weight', 0.5))
        self.pseudo_horizon_ratio = float(getattr(configs, 'gausswarp_pseudo_horizon_ratio', 0.1))
        dropout = float(configs.dropout)
        if self.dimension < 16 or self.dimension % self.num_heads != 0:
            raise ValueError('gausswarp_dimension must be >=16 and divisible by gausswarp_num_heads')
        if self.num_slots < 4 or self.num_layers < 1:
            raise ValueError('GaussWarp requires at least four slots and one mixer layer')
        self.variable_embedding = nn.Embedding(self.num_variables, self.dimension)
        self.slot_embedding = nn.Embedding(self.num_slots, self.dimension)
        centers = torch.linspace(0.0, 1.0, self.num_slots)
        self.register_buffer('base_centers', centers)
        initial_width = max(2.0 / max(self.num_slots - 1, 1), 0.08)
        self.center_offset = nn.Parameter(torch.zeros(self.num_slots))
        self.slot_log_width = nn.Parameter(torch.full((self.num_slots,), math.log(initial_width)))
        self.density_blend_logit = nn.Parameter(torch.full((self.num_variables,), 0.85))
        self.statistic_projection = _mlp(self.num_statistics, 2 * self.dimension, self.dimension, dropout)
        slot_layer = nn.TransformerEncoderLayer(d_model=self.dimension, nhead=self.num_heads, dim_feedforward=3 * self.dimension, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        variable_layer = nn.TransformerEncoderLayer(d_model=self.dimension, nhead=self.num_heads, dim_feedforward=2 * self.dimension, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.slot_mixers = nn.ModuleList([nn.TransformerEncoder(slot_layer, num_layers=1, norm=nn.LayerNorm(self.dimension)) for _ in range(self.num_layers)])
        self.early_variable_mixer = nn.TransformerEncoder(variable_layer, num_layers=1, norm=nn.LayerNorm(self.dimension))
        self.query_encoder = _mlp(1, self.dimension, self.dimension, dropout)
        self.query_reader = nn.MultiheadAttention(self.dimension, self.num_heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(self.dimension)
        self.state_context_encoder = _mlp(self.dimension + 8, 2 * self.dimension, self.dimension, dropout)
        self.context_norm = nn.LayerNorm(self.dimension)
        initial_rates = torch.logspace(-0.3, 1.7, self.num_lag_scales)
        self.lag_log_rates = nn.Parameter(initial_rates.log())
        self.lag_projection = _mlp(5 * self.num_lag_scales, 2 * self.dimension, self.dimension, dropout)
        self.coordinate_gate = nn.Sequential(nn.Linear(2 * self.dimension, self.dimension), nn.GELU(), nn.Linear(self.dimension, self.dimension), nn.Sigmoid())
        self.decoder_input_dimension = 3 * self.dimension + 12
        self.decoder = nn.Sequential(nn.Linear(self.decoder_input_dimension, 2 * self.dimension), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * self.dimension, self.dimension), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.dimension, 6))
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.05)
            self.decoder[-1].bias.copy_(torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]))

    @staticmethod
    def _query_cutoff(y_mark: Tensor, y_mask: Tensor) -> Tensor:
        valid = y_mask.to(torch.bool).any(dim=-1)
        times = y_mark[:, :, 0]
        cutoff = times.masked_fill(~valid, torch.inf).amin(dim=1)
        return torch.where(torch.isfinite(cutoff), cutoff, times.amin(dim=1))

    @staticmethod
    def _masked_location_scale(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        weight = mask.to(x.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        location = (x * weight).sum(dim=1) / count
        centered = (x - location.unsqueeze(1)) * weight
        scale = torch.sqrt(centered.square().sum(dim=1) / count + 1e-05)
        return (location, scale.clamp_min(0.001))

    def _warp_query_time(self, state: dict[str, Tensor], physical_query_time: Tensor) -> Tensor:
        event_count = state['event_mask'].sum(dim=-1)
        query_rank = (event_count > 1).to(dtype=physical_query_time.dtype)
        blend = torch.sigmoid(self.density_blend_logit).view(1, 1, -1)
        return blend * physical_query_time.unsqueeze(-1) + (1.0 - blend) * query_rank.unsqueeze(1)

    def compute_local_weights(self, warped_time: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        slots = self.num_slots
        dtype = warped_time.dtype
        zeta = warped_time.unsqueeze(2)
        max_shift = 0.45 / max(slots - 1, 1)
        centers = (self.base_centers + max_shift * torch.tanh(self.center_offset)).clamp(0.0, 1.0)
        widths = self.slot_log_width.exp().clamp(0.03, 1.0)
        displacement = zeta - centers.view(1, 1, slots, 1)
        weights = torch.exp(-0.5 * (displacement / widths.view(1, 1, slots, 1)).square())
        weights = weights * mask.unsqueeze(2).to(dtype)
        return (weights, centers, widths)

    def _encode(self, x: Tensor, x_mark: Tensor, x_mask: Tensor, cutoff: Tensor) -> dict[str, Tensor]:
        batch_size, steps, variables = x.shape
        raw_time = torch.nan_to_num(x_mark[:, :, 0])
        causal_mask = x_mask.to(torch.bool) & (raw_time <= cutoff.unsqueeze(1) + 1e-07).unsqueeze(-1)
        valid_step = causal_mask.any(dim=-1)
        first_time = raw_time.masked_fill(~valid_step, torch.inf).amin(dim=1)
        first_time = torch.where(torch.isfinite(first_time), first_time, cutoff - 1.0)
        span = (cutoff - first_time).clamp_min(0.0001)
        normalized_time = (raw_time - first_time.unsqueeze(1)) / span.unsqueeze(1)
        location, scale = self._masked_location_scale(x, causal_mask)
        normalized_value = (x - location.unsqueeze(1)) / scale.unsqueeze(1)
        time_by_variable = normalized_time.unsqueeze(1).expand(-1, variables, -1)
        value_by_variable = normalized_value.permute(0, 2, 1)
        raw_value_by_variable = x.permute(0, 2, 1)
        mask_by_variable = causal_mask.permute(0, 2, 1)
        order = time_by_variable.masked_fill(~mask_by_variable, torch.inf).argsort(dim=-1, stable=True)
        time = time_by_variable.gather(-1, order)
        value = value_by_variable.gather(-1, order)
        raw_value = raw_value_by_variable.gather(-1, order)
        mask = mask_by_variable.gather(-1, order)
        time = torch.where(mask, time, torch.zeros_like(time))
        value = torch.where(mask, value, torch.zeros_like(value))
        raw_value = torch.where(mask, raw_value, torch.zeros_like(raw_value))
        lengths = mask.sum(dim=-1)
        rank = torch.arange(steps, device=x.device, dtype=x.dtype).view(1, 1, steps)
        rank = rank / (lengths.to(x.dtype).unsqueeze(-1) - 1.0).clamp_min(1.0)
        blend = torch.sigmoid(self.density_blend_logit).view(1, -1, 1)
        warped_time = blend * time + (1.0 - blend) * rank
        slots = self.num_slots
        weights, centers, _ = self.compute_local_weights(warped_time, mask)
        raw_slot_mass = weights.sum(dim=-1)
        mass = raw_slot_mass.clamp_min(1e-06)
        mean = (weights * value.unsqueeze(2)).sum(dim=-1) / mass
        time_mean = (weights * warped_time.unsqueeze(2)).sum(dim=-1) / mass
        centered_time = warped_time.unsqueeze(2) - time_mean.unsqueeze(-1)
        centered_value = value.unsqueeze(2) - mean.unsqueeze(-1)
        denominator = (weights * centered_time.square()).sum(dim=-1).clamp_min(1e-05)
        slope = ((weights * centered_time * centered_value).sum(dim=-1) / denominator).clamp(-self.slope_limit, self.slope_limit)
        spread = torch.sqrt((weights * centered_value.square()).sum(dim=-1) / mass + 1e-06)
        statistics_parts = [mean, slope / max(self.slope_limit, 1e-06), spread]
        statistics_parts += [torch.log1p(mass), time_mean]
        statistics = torch.stack(statistics_parts, dim=-1)
        variable_embedding = self.variable_embedding(torch.arange(variables, device=x.device))
        slot_embedding = self.slot_embedding(torch.arange(slots, device=x.device))
        tensor = self.statistic_projection(statistics) + variable_embedding.view(1, variables, 1, self.dimension) + slot_embedding.view(1, 1, slots, self.dimension)
        for slot_mixer in self.slot_mixers:
            tensor = slot_mixer(tensor.reshape(batch_size * variables, slots, self.dimension)).reshape(batch_size, variables, slots, self.dimension)
        last_index = (lengths - 1).clamp_min(0)
        previous_index = (lengths - 2).clamp_min(0)
        last_value = value.gather(-1, last_index.unsqueeze(-1)).squeeze(-1)
        previous_value = value.gather(-1, previous_index.unsqueeze(-1)).squeeze(-1)
        last_time = time.gather(-1, last_index.unsqueeze(-1)).squeeze(-1)
        previous_time = time.gather(-1, previous_index.unsqueeze(-1)).squeeze(-1)
        available = lengths > 0
        velocity = ((last_value - previous_value) / (last_time - previous_time).clamp_min(0.001)).clamp(-self.slope_limit, self.slope_limit)
        velocity = torch.where(lengths > 1, velocity, torch.zeros_like(velocity))
        raw_count = lengths.to(x.dtype)
        raw_mean = raw_value.sum(dim=-1) / raw_count.clamp_min(1.0)
        raw_centered = (raw_value - raw_mean.unsqueeze(-1)) * mask.to(x.dtype)
        raw_std = torch.sqrt(raw_centered.square().sum(dim=-1) / raw_count.clamp_min(1.0) + 1e-05)
        raw_last = raw_value.gather(-1, last_index.unsqueeze(-1)).squeeze(-1)
        raw_previous = raw_value.gather(-1, previous_index.unsqueeze(-1)).squeeze(-1)
        raw_velocity = ((raw_last - raw_previous) / (last_time - previous_time).clamp_min(0.001)).clamp(-self.slope_limit, self.slope_limit)
        raw_velocity = torch.where(lengths > 1, raw_velocity, torch.zeros_like(raw_velocity))
        age_to_cutoff = (1.0 - last_time).clamp(0.0, 4.0)
        raw_last = torch.where(available, raw_last, torch.zeros_like(raw_last))
        raw_mean = torch.where(available, raw_mean, torch.zeros_like(raw_mean))
        raw_std = torch.where(available, raw_std, torch.zeros_like(raw_std))
        absolute_summary = torch.stack([raw_last.clamp(-8.0, 8.0) / 8.0, raw_mean.clamp(-8.0, 8.0) / 8.0, raw_std.clamp(0.0, 8.0) / 8.0, raw_velocity / max(self.slope_limit, 1e-06), torch.log1p(raw_count) / math.log1p(float(steps)), available.to(x.dtype), age_to_cutoff / 4.0], dim=-1)
        return {'memory': tensor, 'slot_mean': mean, 'slot_slope': slope, 'slot_spread': spread, 'slot_log_mass': torch.log1p(raw_slot_mass), 'centers': centers, 'last_value': torch.where(available, last_value, torch.zeros_like(last_value)), 'last_time': torch.where(available, last_time, torch.zeros_like(last_time)), 'velocity': velocity, 'available': available, 'location': location, 'scale': scale, 'first_time': first_time, 'span': span, 'event_time': time, 'event_value': value, 'event_mask': mask, 'slot_time_mean': time_mean, 'absolute_summary': absolute_summary}

    @staticmethod
    def _evidence_conditioned_transport_scale(parameters: Tensor, variable_scale: Tensor, local_scale: Tensor, shared_scale: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scale_logits = parameters[..., 2:5]
        scale_weights = torch.softmax(scale_logits, dim=-1)
        candidates = torch.stack([variable_scale, local_scale, shared_scale], dim=-1)
        mixture = (scale_weights * candidates).sum(dim=-1)
        gain = torch.exp(0.5 * torch.tanh(parameters[..., 5]))
        return (mixture * gain, scale_weights, gain)

    def _simple_state_context(self, state: dict[str, Tensor], lead: Tensor) -> Tensor:
        batch_size, queries, variables = lead.shape
        variable_embedding = self.variable_embedding(torch.arange(variables, device=lead.device))
        absolute_summary = state['absolute_summary'].unsqueeze(1).expand(-1, queries, -1, -1)
        variable_context = variable_embedding.view(1, 1, variables, -1).expand(batch_size, queries, -1, -1)
        query_numeric = torch.stack([lead], dim=-1)
        state_input = torch.cat([absolute_summary, variable_context, query_numeric], dim=-1)
        return self.state_context_encoder(state_input)

    def _lag_moments(self, state: dict[str, Tensor], query_time: Tensor) -> Tensor:
        lag = (query_time.unsqueeze(-1).unsqueeze(-1) - state['event_time'].unsqueeze(1)).clamp_min(0.0)
        mask = state['event_mask'].unsqueeze(1).to(query_time.dtype)
        value = state['event_value'].unsqueeze(1)
        rates = torch.sort(self.lag_log_rates.exp()).values.clamp(0.1, 100.0)
        weight = torch.exp(-lag.unsqueeze(-1) * rates.view(1, 1, 1, 1, -1))
        weight = weight * mask.unsqueeze(-1)
        mass = weight.sum(dim=3).clamp_min(1e-06)
        mean = (weight * value.unsqueeze(-1)).sum(dim=3) / mass
        mean_lag = (weight * lag.unsqueeze(-1)).sum(dim=3) / mass
        centered_lag = lag.unsqueeze(-1) - mean_lag.unsqueeze(3)
        centered_value = value.unsqueeze(-1) - mean.unsqueeze(3)
        lag_variance = (weight * centered_lag.square()).sum(dim=3).clamp_min(1e-05)
        slope = (weight * centered_lag * centered_value).sum(dim=3) / lag_variance
        intercept = mean - slope * mean_lag
        spread = torch.sqrt((weight * centered_value.square()).sum(dim=3) / mass + 1e-06)
        features = torch.stack([mean, intercept, spread, torch.log1p(mass), mean_lag], dim=-1).flatten(-2)
        return self.lag_projection(features)

    def _local_slot_summary(self, state: dict[str, Tensor], warped_query_time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        selection_distance = warped_query_time.unsqueeze(-1) - state['centers'].view(1, 1, 1, -1)
        local_weight = torch.softmax(-0.5 * selection_distance.square(), dim=-1)
        regression_distance = warped_query_time.unsqueeze(-1) - state['slot_time_mean'].unsqueeze(1)
        local_path = (local_weight * (state['slot_mean'].unsqueeze(1) + state['slot_slope'].unsqueeze(1) * regression_distance)).sum(dim=-1)
        local_spread = (local_weight * state['slot_spread'].unsqueeze(1)).sum(dim=-1)
        local_log_mass = (local_weight * state['slot_log_mass'].unsqueeze(1)).sum(dim=-1)
        return (local_path, local_spread, local_log_mass)

    def _decode(self, state: dict[str, Tensor], y_mark: Tensor) -> Tensor:
        batch_size, queries = y_mark.shape[:2]
        variables = self.num_variables
        physical_query_time = (y_mark[:, :, 0] - state['first_time'].unsqueeze(1)) / state['span'].unsqueeze(1)
        warped_query_time = self._warp_query_time(state, physical_query_time)
        lead = (physical_query_time.unsqueeze(-1) - state['last_time'].unsqueeze(1)).clamp(0.0, 4.0)
        variable_embedding = self.variable_embedding(torch.arange(variables, device=y_mark.device))
        query_features = physical_query_time.unsqueeze(-1).unsqueeze(-1)
        query_features = query_features.expand(-1, -1, variables, -1)
        query = self.query_encoder(query_features) + variable_embedding.view(1, 1, variables, self.dimension)
        lag_context = self._lag_moments(state, physical_query_time)
        coordinate_gate = self.coordinate_gate(torch.cat([query, lag_context], dim=-1))
        temporal_query = coordinate_gate * query + (1.0 - coordinate_gate) * lag_context
        state_context = self._simple_state_context(state, lead)
        enriched_query = self.context_norm(temporal_query + state_context)
        mixed_query = self.early_variable_mixer(enriched_query.reshape(batch_size * queries, variables, self.dimension)).reshape(batch_size, queries, variables, self.dimension)
        flat_query = mixed_query.permute(0, 2, 1, 3).reshape(batch_size * variables, queries, self.dimension)
        flat_memory = state['memory'].reshape(batch_size * variables, self.num_slots, self.dimension)
        slot_context, _ = self.query_reader(flat_query, flat_memory, flat_memory, need_weights=False)
        slot_context = self.query_norm(slot_context + flat_query)
        slot_context = slot_context.reshape(batch_size, variables, queries, self.dimension).permute(0, 2, 1, 3)
        mixed = slot_context
        context = mixed_query
        local_path, local_spread, local_log_mass = self._local_slot_summary(state, warped_query_time)
        last = state['last_value'].unsqueeze(1).expand(-1, queries, -1)
        transported_local = local_path
        velocity = state['velocity'].unsqueeze(1).expand_as(last)
        available = state['available'].to(y_mark.dtype).unsqueeze(1).expand_as(last)
        variable_scale = state['scale'].unsqueeze(1).expand_as(last)
        availability = state['available'].to(y_mark.dtype)
        shared_scale = torch.sqrt((state['scale'].square() * availability).sum(dim=-1) / availability.sum(dim=-1).clamp_min(1.0) + 1e-06).unsqueeze(-1).unsqueeze(-1).expand_as(last)
        local_scale = variable_scale * (0.5 + 0.5 * local_spread.clamp(0.0, 3.0))
        shared_to_variable = torch.log(shared_scale / variable_scale.clamp_min(0.001)).clamp(-4.0, 4.0)
        numeric_values = [last, transported_local, velocity / max(self.slope_limit, 1e-06), lead, torch.log1p(lead), available, state['scale'].log().unsqueeze(1).expand_as(last), physical_query_time.unsqueeze(-1).expand_as(last), (physical_query_time - 1.0).unsqueeze(-1).expand_as(last), torch.log1p(local_spread.clamp_min(0.0)), local_log_mass / math.log1p(float(state['event_time'].shape[-1])), shared_to_variable]
        numeric = torch.stack(numeric_values, dim=-1)
        decoder_parts = [mixed, context, variable_embedding.view(1, 1, variables, -1).expand(batch_size, queries, -1, -1), numeric]
        decoder_input = torch.cat(decoder_parts, dim=-1)
        parameters = self.decoder(decoder_input)
        location = state['location'].unsqueeze(1)
        decoder_gate = torch.sigmoid(parameters[..., 1])
        base = decoder_gate * last + (1.0 - decoder_gate) * transported_local
        transport_scale, _, _ = self._evidence_conditioned_transport_scale(parameters, variable_scale, local_scale, shared_scale)
        base_raw = base * variable_scale + location
        innovation = self.residual_limit * torch.tanh(parameters[..., 0])
        return base_raw + transport_scale * innovation

    def forward(self, x: Tensor, x_mark: Tensor, x_mask: Tensor | None, **kwargs) -> dict:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        y, y_mark, y_mask = (kwargs['y'], kwargs['y_mark'], kwargs['y_mask'])
        query_mask = kwargs.get('query_mask', y_mask)
        cutoff = self._query_cutoff(y_mark, query_mask)
        state = self._encode(x, x_mark, x_mask, cutoff)
        prediction = self._decode(state, y_mark)
        auxiliary_loss = prediction.sum() * 0.0
        if kwargs.get('exp_stage') == 'train' and self.auxiliary_weight > 0:
            fraction = 0.55 + 0.3 * torch.rand_like(cutoff)
            full_span = (cutoff - state['first_time']).clamp_min(0.0001)
            pseudo_cutoff = state['first_time'] + fraction * full_span
            pseudo_state = self._encode(x, x_mark, x_mask, pseudo_cutoff)
            pseudo_prediction = self._decode(pseudo_state, x_mark)
            history_time = x_mark[:, :, 0]
            upper = pseudo_cutoff + self.pseudo_horizon_ratio * full_span
            pseudo_mask = (x_mask.to(torch.bool) & (history_time > pseudo_cutoff.unsqueeze(1) + 1e-07).unsqueeze(-1) & (history_time <= upper.unsqueeze(1) + 1e-07).unsqueeze(-1) & (history_time <= cutoff.unsqueeze(1) + 1e-07).unsqueeze(-1)).to(x.dtype)
            auxiliary_loss = ((pseudo_prediction - x).square() * pseudo_mask).sum() / pseudo_mask.sum().clamp_min(1.0)
        feature_start = -1 if self.configs.features == 'MS' else 0
        return {'pred': prediction[:, :, feature_start:], 'true': y[:, :, feature_start:], 'mask': y_mask[:, :, feature_start:] if y_mask is not None else None, 'auxiliary_loss': auxiliary_loss}
