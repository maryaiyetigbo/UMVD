import torch
import torch.nn.functional as F


def loss_function(output, gt, mode="mse", device="cpu"):
    if(mode == "mse"):
        loss = F.mse_loss(output, gt, reduction="sum") / (gt.size(0) * 2)
        
    if(mode =="l1"):
        loss = torch.nn.functional.l1_loss(output, gt, reduction="sum")
    return loss