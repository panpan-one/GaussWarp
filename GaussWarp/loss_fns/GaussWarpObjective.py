import torch
import torch.nn as nn

class Loss(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.auxiliary_weight = float(configs.gausswarp_auxiliary_weight)

    def forward(self, pred, true, mask=None, auxiliary_loss=None, exp_stage=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(true)
        mask = mask.to(dtype=pred.dtype)
        count = mask.sum().clamp_min(1.0)
        residual = (pred - true) * mask
        forecast = residual.square().sum() / count
        if exp_stage != 'train':
            return {'loss': forecast, 'forecast_loss': forecast, 'auxiliary_loss': forecast.new_zeros(())}
        if auxiliary_loss is None:
            auxiliary_loss = forecast.new_zeros(())
        total = forecast + self.auxiliary_weight * auxiliary_loss
        return {'loss': total, 'forecast_loss': forecast, 'auxiliary_loss': auxiliary_loss}
