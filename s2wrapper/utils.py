#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch

def split_chessboard(x, num_split):
    """
        x: Tensor of (b * c * h * w)
        num_split: (num_split_h, num_split_w)
        Deividing x into num_split_h and num_split_w parts on height and width dimenstion, separately, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    num_split_h, num_split_w = num_split
    assert H % num_split_h == 0 and W % num_split_w == 0
    h, w = H // num_split_h, W // num_split_w
    x_split = torch.cat([x[:, :, i*h:(i+1)*h, j*w:(j+1)*w] for i in range(num_split_h) for j in range(num_split_w)], dim=0)
    return x_split

def merge_chessboard(x, num_split):
    """
        x: Tensor of (b * c * h * w)
        num_split: (num_split_h, num_split_w)
        Assuming x contains (num_split_h * num_split_w) sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        (inverse of split_chessboard)
    """
    B, C, H, W = x.shape
    num_split_h, num_split_w = num_split
    assert B % (num_split_h * num_split_w) == 0
    b = B // (num_split_h * num_split_w)
    x_merge = torch.cat([torch.cat([x[(i*num_split_w + j)*b:(i*num_split_w + j + 1)*b] for j in range(num_split_w)], dim=-1)
                         for i in range(num_split_h)], dim=-2)
    return x_merge

def batched_forward(model, x, batch_size=-1):
    if batch_size == -1:
        return model(x)
    else:
        x_batched = x.split(batch_size)
        outs = [model(x) for x in x_batched]
        return torch.cat(outs, dim=0)

