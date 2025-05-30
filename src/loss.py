import torch, torch.nn as nn


mse = nn.MSELoss()
def mse_loss(pred, tgt):  return mse(pred, tgt)
def rmse_loss(pred, tgt): return torch.sqrt(mse(pred, tgt))

def l1_regularizer(model: nn.Module):
    return sum(p.abs().sum() for p in model.parameters())
