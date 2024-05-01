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
from .utils import split_chessboard, merge_chessboard, batched_forward

def forward(model, input, scales=None, img_sizes=None, max_split_size=None, split_mode="resize", resize_output_to_idx=0, num_prefix_token=0,
            output_shape='bnc', split_forward=False):

    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert all([isinstance(size, int) for size in img_sizes]) or all([isinstance(size, tuple) for size in img_sizes]), \
        "img_sizes should be a list of integers (for square images) or a list of tuples (for non-square images)."
    assert split_mode in ['resize', 'pad'], "split_mode should be either 'resize' or 'pad'."
    assert output_shape in ['bnc', 'bchw'], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
    assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

    b, c, input_h, input_w = input.shape

    # image size for each scale
    assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
    if img_sizes is None:
        img_sizes = [(int(input_h * scale), int(input_w * scale)) for scale in scales]
    elif isinstance(img_sizes[0], int):
        img_sizes = [(size, size) for size in img_sizes]
    
    # interpolate image to each size
    input_multiscale = [F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype) for size in img_sizes]

    # resize or pad input image to make it divisible by max_split_size
    assert max_split_size is not None or input_h == input_w, "max_split_size should be assigned for non-square images."
    max_split_size = max_split_size or input_h   # The maximum size of each split of image. Set as the input size by default
    num_splits = [(math.ceil(size[0] / max_split_size), math.ceil(size[1] / max_split_size)) for size in img_sizes]   # number of splits each scale
    original_img_sizes = img_sizes
    img_sizes = [(num_split[0] * max_split_size, num_split[1] * max_split_size) for num_split in num_splits]
    if split_mode == 'resize':
        input_multiscale = [F.interpolate(x.to(torch.float32), size=size, mode='bicubic').to(x.dtype) for size, x in zip(img_sizes, input_multiscale)]
    elif split_mode == 'pad':
        input_multiscale = [F.pad(x, (0, size[1] - x.shape[-1], 0, size[0] - x.shape[-2])) for size, x in zip(img_sizes, input_multiscale)]

    # prepare multiscale inputs
    input_multiscale = [split_chessboard(x, num_split=num_split) for num_split, x in zip(num_splits, input_multiscale)]

    # run feedforward on each scale
    outs_multiscale = [batched_forward(model, x, b) if split_forward else model(x) for x in input_multiscale]
    if num_prefix_token > 0:
        outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
    if output_shape == 'bnc':
        outs_multiscale = [rearrange(out, 'b (h w) c -> b c h w', h=int(out.shape[1] ** 0.5), w=int(out.shape[1] ** 0.5))
                           for out in outs_multiscale]

    # merge outputs of different splits for each scale separately
    outs_multiscale = [merge_chessboard(out, num_split=num_split) for num_split, out in zip(num_splits, outs_multiscale)]

    # resize or unpad the output features to approximately the original size
    original_feature_sizes = [(math.ceil(original_img_size[0] / img_size[0] * feature_size[0]), math.ceil(original_img_size[1] / img_size[1] * feature_size[1]))
                              for original_img_size, img_size, feature_size in zip(original_img_sizes, img_sizes, [out.shape[-2:] for out in outs_multiscale])]
    if split_mode == 'resize':
        outs_multiscale = [F.interpolate(out.to(torch.float32), size=original_size, mode='area').to(out.dtype) for original_size, out in zip(original_feature_sizes, outs_multiscale)]
    elif split_mode == 'pad':
        outs_multiscale = [out[:, :, :original_size[0], :original_size[1]] for original_size, out in zip(original_feature_sizes, outs_multiscale)]

    # interpolate outputs from different scales and concat together
    output_size = outs_multiscale[resize_output_to_idx].shape[-2:]
    out = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                   mode='area').to(outs_multiscale[i].dtype)
                     for i in range(len(outs_multiscale))], dim=1)
    if output_shape == 'bnc':
        out = rearrange(out, 'b c h w -> b (h w) c')
    if num_prefix_token > 0:
        # take the mean of prefix tokens from different splits for each scale
        outs_prefix_multiscale = [torch.stack(out.split(b, dim=0), dim=0).mean(dim=0) for out in outs_prefix_multiscale]
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out
