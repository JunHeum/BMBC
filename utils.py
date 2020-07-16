import time
import torch
import torch.nn as nn

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is  "+ str(time.time() - startTime_for_tictoc)+"  seconds")
    else:
        print("Toc: start time not set")

def warp(x,flo, return_mask=False):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask = mask.masked_fill_(mask < 0.999, 0)
    mask = mask.masked_fill_(mask > 0, 1)

    if return_mask:
        return output * mask, mask
    else:
        return output * mask