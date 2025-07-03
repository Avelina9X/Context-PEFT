import logging
import os
import gc
import math
import dataclasses
from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Any

import hashlib
import yaml

import wandb
import tqdm

import torch
from torch import nn
from torcheval import metrics

from transformers import CLIPVisionModel, AutoImageProcessor, AutoTokenizer, AutoConfig, BatchFeature
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.cache_utils import StaticCache

from model import ContextPeftConfig, ContextPeftProcessor, ContextPeftForConditionalGeneration
from model.modeling_context_peft import CONTEXT_PEFT_WRAPPER_MAPPING

from data import BaseDataset, CocoDataset, compute_f1

from .trainer_config import TrainerConfig
from .lr_schedules import SCHEDULE_MAP

def seed_hash( string: str, offset: int ) -> int:
    """ Generates an int32 seed from a string and int.

    Args:
        string (str): String to seed from
        offset (int): Numerical identifier 

    Returns:
        int: sha1 digest of "{string}_{offset}"
    """
    sha1 = hashlib.sha1()
    sha1.update( str.encode( f'{string}_{offset}' ) )
    sha1_hex = sha1.hexdigest()
    return int( sha1_hex, 16 ) % 4294967295

def get_adaptors( task: str, context: str | None, num_hidden_layers: int, peft_type: str | None, lora_image_scale: float | None ):
    if context is None:
        return None
    
    adaptors = {}
    contexts = {
        'image': [ 'image' ],
        'text': [ 'text' ],
        'both': [ 'image', 'text' ],
        'shared': [ 'shared' ]
    }

    for c in contexts[context]:
        if c == 'image':
            last = num_hidden_layers - 1
            adaptors[ f'{task}:image' ] = {
                'context': 'image',
                'exclude_modules': [
                    f'layers.{last}.self_attn.q_proj',
                    f'layers.{last}.self_attn.o_proj',
                    f'layers.{last}.mlp',
                ]
            }
            if peft_type == 'lora':
                adaptors[ f'{task}:image' ][ 'scale_multiplier' ] = lora_image_scale
                
        elif c == 'text':
            adaptors[ f'{task}:text' ] = {
                'context': 'text'
            }                
        elif c == 'shared':
            adaptors[ f'{task}:shared' ] = {
                'context': [ 'text', 'image' ]
            }
        else:
            raise ValueError( f'Invalid context {c}!' )

    return adaptors

def get_peft_config( peft_type: str | None, adaptor_kwargs: dict | None ):
    adaptor_kwargs = adaptor_kwargs or {}

    if peft_type is None:
        return None
    elif peft_type == 'lora':
        config = {
            'type': 'lora',
            'target_modules': [ 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj' ],
            'exclude_modules': None,
            'use_bias': 'auto',
            'scale_multiplier': 1.0,
        }
    elif peft_type == 'bitfit':
        config = {
            'type': 'bitfit',
            'target_modules': [ 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj' ],
            'exclude_modules': None,
            'force_bias': False,
        }
    elif peft_type == 'ia3':
        config = {
            'type': 'ia3',
            'target_modules': [ 'k_proj', 'v_proj', 'down_proj' ],
            'exclude_modules': None,
            'feedforward_modules': [ 'down_proj' ]
        }
    else:
        raise ValueError( f'Invalid peft type {peft_type}!' )

    config.update( **adaptor_kwargs )

    return config