import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers import UNet2DConditionModel
from diffusers.models.lora import LoRALinearLayer

from transformers import CLIPTextModel
from diffusers.loaders import LoraLoaderMixin
import os

class LoraLayerNorm(nn.Module):

    def __init__(self, normalized_shape, weight, bias, eps=1e-5):
        super(LoraLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(weight) if weight is not None else None
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        ###
        self.lora_weight = torch.nn.Parameter(torch.randn(normalized_shape ) *0.02)
        self.lora_bias = torch.nn.Parameter(torch.zeros(normalized_shape ) *0.02)

    def forward(self, x):
        weight = self.weight.to(self.lora_weight.dtype) + self.lora_weight
        bias = self.bias.to(self.lora_weight.dtype) + self.lora_bias

        orig_dtype = x.dtype

        x = F.layer_norm(
            x, self.normalized_shape, weight, bias, self.eps
        )

        x = x.to(orig_dtype)

        return x


class LoraGroupNorm(nn.Module):

    def __init__(self, num_groups, num_channels, weight=None, bias=None, eps=1e-05, affine=True):
        super(LoraGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = None

        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

        ###
        self.lora_weight = torch.nn.Parameter(torch.randn(num_channels ) *0.02)
        self.lora_bias = torch.nn.Parameter(torch.zeros(num_channels ) *0.02)

    def forward(self, x):

        orig_dtype = x.dtype

        weight = self.weight.to(self.lora_weight.dtype) + self.lora_weight
        bias = self.bias.to(self.lora_weight.dtype) + self.lora_bias

        x = F.group_norm(
            x, self.num_groups, weight, bias, self.eps
        )

        x = x.to(orig_dtype)

        return x


def patch_lora_norms(model):
    names = []
    unet_target_names = ["norm", "norm1", "norm2", "norm3", "conv_norm_out"]
    text_encoder_names = ["layer_norm1", "layer_norm2", "final_layer_norm"]
    target_names = unet_target_names + text_encoder_names
    for name, module in model.named_modules():
        has = [hasattr(module, target_name) for target_name in target_names]
        if any(has):
            for index in np.where(np.array(has))[0]:
                norm_name = target_names[index]
                old_norm = getattr(module, norm_name)
                weight = old_norm.weight.data.clone()
                bias = old_norm.bias.data.clone()
                eps = old_norm.eps

                if isinstance(old_norm, nn.LayerNorm):
                    normalized_shape = old_norm.normalized_shape
                    new_norm = LoraLayerNorm(
                        normalized_shape,
                        weight,
                        bias,
                        eps
                    )
                elif isinstance(old_norm, nn.GroupNorm):
                    new_norm = LoraGroupNorm(
                        old_norm.num_groups,
                        old_norm.num_channels,
                        weight,
                        bias,
                        eps,
                        old_norm.affine
                    )

                delattr(module, norm_name)

                module.add_module(norm_name, new_norm)


def set_lora_attn(model, rank):
    for attn_processor_name, attn_processor in model.attn_processors.items():
        # Parse the attention module.
        attn_module = model
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=rank
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=rank
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=rank
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=rank,
            )
        )


def load_pipeline_with_lora_norm(pipeline_class, lora_path, pretrained_model_path):

    pipeline = pipeline_class.from_pretrained(pretrained_model_path)

    # load weights
    weights = torch.load(lora_path)

    norm_only = True
    for k in weights.keys():
        if not "norm" in k.lower():
            norm_only = False
            break

    # check if we have text encoder weights
    has_text_encoder =  True if weights["text_encoder"] is not None else False

    # patch
    patch_lora_norms(pipeline.unet)

    if not norm_only:
        # determine rank
        for name, weight in weights["unet"].items():
            if "lora" in name:
                unet_rank = min(weight.shape)
                break

        set_lora_attn(pipeline.unet, unet_rank)

    pipeline.unet.load_state_dict(weights["unet"], strict=False)

    if has_text_encoder:
        # get rank
        if not norm_only:
            for name, weight in weights["text_encoder"].items():
                if "lora" in name:
                    text_encoder_rank = min(weight.shape)
                    break
            LoraLoaderMixin._modify_text_encoder(pipeline.text_encoder, dtype=torch.float32, rank=text_encoder_rank)

        patch_lora_norms(pipeline.text_encoder)
        pipeline.text_encoder.load_state_dict(weights["text_encoder"], stict=False)


    return pipeline
