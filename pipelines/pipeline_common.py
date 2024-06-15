import torch
import gc
from torch import nn
#from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
import accelerate
import bitsandbytes as bnb

def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
   
    gc.collect()

#copied and adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_utils.py
def optionally_disable_offloading(_pipeline):

    print(
            fr"Restarting CPU Offloading for {_pipeline.transformer_name}..."
          )
    for _, model in _pipeline.components.items():
        if isinstance(model, torch.nn.Module) and hasattr(model, "_hf_hook"):
            accelerate.hooks.remove_hook_from_module(model, recurse=True)
    _pipeline._all_hooks = []


def quantize_4bit(module, dtype):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            device = child.weight.data.device

            # Create and configure the Linear layer
            has_bias = True if child.bias is not None else False
            
            # TODO: Make that configurable
            # fp16 for compute dtype leads to faster inference
            # and one should almost always use nf4 as a rule of thumb
            bnb_4bit_compute_dtype = dtype
            quant_type = "nf4"

            new_layer = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=has_bias,
                compute_dtype=bnb_4bit_compute_dtype,
                quant_type=quant_type,
            )

            new_layer.load_state_dict(child.state_dict())
            new_layer = new_layer.to(device)

            # Set the attribute
            setattr(module, name, new_layer)
        else:
            # Recursively apply to child modules
            quantize_4bit(child, dtype)

def quantize_8bit(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            device = child.weight.data.device

            # Create and configure the Linear layer
            has_bias = True if child.bias is not None else False
            
            new_layer = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=has_bias
            )

            new_layer.load_state_dict(child.state_dict())
            new_layer = new_layer.to(device)

            # Set the attribute
            setattr(module, name, new_layer)
        else:
            # Recursively apply to child modules
            quantize_8bit(child)