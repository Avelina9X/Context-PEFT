import torch
from torch import nn

from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from transformers import CLIPVisionModel, AutoImageProcessor, AutoTokenizer, AutoConfig, PretrainedConfig, BaseImageProcessor, PreTrainedTokenizerBase

from model.configuration_context_peft import ContextPeftConfig
from model.processing_context_peft import ContextPeftProcessor
from model.modeling_context_peft import ContextPeftForConditionalGeneration, CONTEXT_PEFT_WRAPPER_MAPPING
from data.coco import CocoDataset

from data.base_dataset import BaseDataset

from .trainer_config import TrainerConfig

def get_adaptors( task: str, context: str | None, num_hidden_layers: int ):
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

def get_peft_config( peft_type: str | None, lora_rank: int | None ):
    if lora_rank and peft_type != 'lora':
        raise ValueError( 'Cannot specify a lora rank when peft type isn\'t lora!' )

    if peft_type is None:
        return None
    elif peft_type == 'lora':
        return {
            'type': 'lora',
            'r': lora_rank,
            'target_modules': [ 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj' ],
            'exclude_modules': None,
            'lora_alpha': lora_rank,
            'use_rslora': False,
            'use_bias': 'auto',
        }
    elif peft_type == 'bitfit':
        return {
            'type': 'bitfit',
            'target_modules': [ 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj' ],
            'exclude_modules': None,
            'force_bias': False,
        }
    elif peft_type == 'ia3':
        return {
            'type': 'ia3',
            'target_modules': [ 'k_proj', 'v_proj', 'down_proj' ],
            'exclude_modules': None,
            'feedforward_modules': [ 'down_proj' ]
        }
    else:
        raise ValueError( f'Invalid peft type {peft_type}!' )

class Trainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
    ):
        self.trainer_config = trainer_config

        if trainer_config.stage == 'stage1':
            self.load_pipeline_stage1()
        else:
            raise ValueError( 'stage2 not yet implemented!' )

        self.get_dataset()

    def load_pipeline_stage1( self ):
        vision_model_name = self.trainer_config.vision_model_name
        text_model_name = self.trainer_config.text_model_name
        assert vision_model_name is not None
        assert text_model_name is not None
        
        vision_config = CLIPVisionModel.from_pretrained( vision_model_name ).config
        vision_processor = AutoImageProcessor.from_pretrained( vision_model_name, use_fast=True )
        image_seq_len = ( vision_config.image_size // vision_config.patch_size ) ** 2 + 1

        text_config = AutoConfig.from_pretrained( text_model_name )
        text_tokenizer = AutoTokenizer.from_pretrained( text_model_name, use_fast=True )

        processor = ContextPeftProcessor(
            image_processor=vision_processor,
            tokenizer=text_tokenizer,
            image_seq_len=image_seq_len,
            chat_template='chat_ml'
        )

        peft_config = get_peft_config(
            self.trainer_config.peft_type,
            self.trainer_config.lora_rank
        )

        adaptors = get_adaptors(
            self.trainer_config.dataset,
            self.trainer_config.adaptor_context,
            text_config.num_hidden_layers
        )

        config = ContextPeftConfig(
            vision_config=vision_config,
            vision_dim=vision_config.hidden_size,
            vision_trainable=False,

            text_config=text_config,
            text_dim=text_config.hidden_size,
            text_trainable=self.trainer_config.text_trainable,

            image_pad_token_id=processor.get_image_pad_token_id(),

            peft_type=self.trainer_config.peft_type,
            default_peft_config=peft_config,
            adaptors=adaptors,

            attn_implementation='flash_attention_2' if torch.cuda.is_available() else 'sdpa'
        )

        model = ContextPeftForConditionalGeneration( config, load_from_hub=True )
        model.train()

        self.processor = processor
        self.model = model
        self.model_config = config
        
    def get_dataset( self ):
        if self.trainer_config.dataset == 'coco':
            dataset = CocoDataset(
                processor=self.processor,
                assistant_prefix='<|im_start|>assistant\n',
                assistant_suffix='<|im_end|>',
                batch_size=self.trainer_config.batch_size,
                sequence_length=self.trainer_config.sequence_length,
                download_timeout=4 * 60 * 60,
            )

            if self.trainer_config.sequence_length == -1:
                dataset.set_optimal_sequence_length( self.trainer_config.pad_to_multiple )
        else:
            raise ValueError( f'Invalid dataset {self.trainer_config.dataset}' )

        self.dataset = dataset

    def get_optimizer( self ):
        ...

    def get_param_groups( self ):
        base_decay = self.trainer_config.weight_decay
        adaptor_decay = self.trainer_config.adaptor_decay

        if isinstance( adaptor_decay, bool ):
            adaptor_decay = base_decay if adaptor_decay else 0.0
        
        adaptor_layer_names: list[str] = []
        for wrapper_class in CONTEXT_PEFT_WRAPPER_MAPPING.values():
            adaptor_class = wrapper_class.adaptor_class
            adaptor_layer_names += adaptor_class.adaptor_layer_names

        adaptor_parameters: list[str] = []
        base_parameters: list[str] = []

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if any( exclude in name for exclude in adaptor_layer_names ):
                    adaptor_parameters.append( name )
                else:
                    base_parameters.append( name )
        
        base_decay_parameters = get_parameter_names( self.model, [ *ALL_LAYERNORM_LAYERS, nn.Embedding ] )
        base_decay_parameters = [ name for name in base_decay_parameters if name in base_parameters ]
        base_decay_parameters = [ name for name in base_decay_parameters if 'bias' not in name ]
        base_decay_parameters = [ name for name in base_decay_parameters if 'norm' not in name ]
        base_decay_parameters = [ name for name in base_decay_parameters if 'class_embedding' not in name ]

        base_nocay_parameters = [ name for name in base_parameters if name not in base_decay_parameters ]

        param_groups = [
            {
                'params': [ p for n, p in self.model.named_parameters() if n in base_decay_parameters ],
                'weight_decay': base_decay,
            },
            {
                'params': [ p for n, p in self.model.named_parameters() if n in base_nocay_parameters ],
                'weight_decay': 0,
            },
            {
                'params': [ p for n, p in self.model.named_parameters() if n in adaptor_parameters ],
                'weight_decay': adaptor_decay,
            },
        ]
        param_groups = [ g for g in param_groups if g[ 'params' ] ]

        return param_groups