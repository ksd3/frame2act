import torch.nn.functional as F


def compute_metrics(pred, target, act_std, act_mean):
    """Returns dict of per-dim and aggregate metrics (unnormalised MAE, MSE)."""
    mse_norm = F.mse_loss(pred, target).item()
    mae_norm = (pred - target).abs().mean().item()
    # unnormalise
    pred_un = pred * act_std + act_mean
    target_un = target * act_std + act_mean
    mse_un = F.mse_loss(pred_un, target_un).item()
    mae_un = (pred_un - target_un).abs().mean().item()
    mae_dim = (pred_un - target_un).abs().mean(dim=0)  # (2,)
    return dict(
        mse_norm=mse_norm, mae_norm=mae_norm,
        mse=mse_un, mae=mae_un,
        mae_steer=mae_dim[0].item(), mae_accel=mae_dim[1].item(),
    )
