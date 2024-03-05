#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from .utils import split_chessboard, merge_chessboard

def forward(model, input, scales=None, img_size_each_scale=None, max_split_size=None, resize_output_to_idx=0, num_prefix_token=0):

    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW!"
    assert input.shape[2] == input.shape[3], "Currently only support square images!"
    b, c, input_size, _ = input.shape

    # image size for each scale
    assert scales is not None or img_size_each_scale is not None, "Must assign either scales or img_size_each_scale!"
    img_size_each_scale = img_size_each_scale or [input_size * scale for scale in scales]

    # prepare multiscale inputs
    max_split_size = max_split_size or input_size   # The maximum size of each split of image. Set as the input size by default
    num_splits = [math.ceil(size / max_split_size) for size in img_size_each_scale]   # number of splits each scale
    input_multiscale = []
    for size, num_split in zip(img_size_each_scale, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # run feedforward on each scale
    outs_multiscale = [model(x) for x in input_multiscale]
    if num_prefix_token > 0:
        outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
    outs_multiscale = [rearrange(out, 'b (h w) c -> b c h w', h=int(out.shape[1] ** 0.5), w=int(out.shape[1] ** 0.5))
                       for out in outs_multiscale]

    # merge outputs of different splits for each scale separately
    outs_multiscale = [merge_chessboard(out, num_split=num_split) for num_split, out in zip(num_splits, outs_multiscale)]

    # interpolate outputs from different scales and concat together
    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                   mode='area').to(outs_multiscale[i].dtype)
                     for i in range(len(outs_multiscale))], dim=1)
    out = rearrange(out, 'b c h w -> b (h w) c')
    if num_prefix_token > 0:
        # take the mean of prefix tokens from different splits for each scale
        outs_prefix_multiscale = [torch.stack(out.split(b, dim=0), dim=0).mean(dim=0) for out in outs_prefix_multiscale]
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out
