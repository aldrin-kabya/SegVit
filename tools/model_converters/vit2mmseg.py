# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_vit(ckpt):
    """
    Convert ViT model weights to MMSegmentation format.

    Args:
        ckpt (dict): The checkpoint dictionary containing model weights.
    """
    # Check if the 'encoder' key exists, if it does, use that part of the checkpoint
    if 'encoder' in ckpt:
        ckpt = ckpt['encoder']  # Extract the ViT model weights from the 'encoder' dictionary

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        # Handle 'module.' prefix if present (which occurs when model was wrapped in nn.DataParallel)
        if k.startswith('module.'):
            k = k.replace('module.', '')

        # Skip the classification head
        if k.startswith('head'):
            continue

        # Convert norm layers
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        # Convert patch embedding layer
        elif k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        # Convert transformer block layers
        elif k.startswith('blocks'):
            if 'norm' in k:
                new_k = k.replace('norm', 'ln')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in k:
                new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = k
            # Replace 'blocks.' with 'layers.' to match MMSegmentation structure
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k

        # Add the converted key-value pair to the new checkpoint dictionary
        new_ckpt[new_k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models or custom pretrained ViT models to MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    # Determine which part of the checkpoint to convert
    if 'state_dict' in checkpoint:
        # timm-style checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        # If 'encoder' exists, it will be handled in convert_vit automatically
        state_dict = checkpoint

    # Convert the ViT model, automatically checking if 'encoder' exists
    weight = convert_vit(state_dict)
    
    # Ensure the destination directory exists
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    # Save the converted weights
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
