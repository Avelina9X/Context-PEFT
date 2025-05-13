import gc
import math
from dataclasses import dataclass
import multiprocessing as mp

import tqdm
import torch
from torch import device, nn

from torcheval import metrics

from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.feature_extraction_utils import BatchFeature

from transformers import CLIPVisionModel, AutoImageProcessor, AutoTokenizer, AutoConfig, PretrainedConfig, BaseImageProcessor, PreTrainedTokenizerBase

from model.configuration_context_peft import ContextPeftConfig
from model.processing_context_peft import ContextPeftProcessor
from model.modeling_context_peft import ContextPeftForConditionalGeneration, CONTEXT_PEFT_WRAPPER_MAPPING
from data.coco import CocoDataset

from data.base_dataset import BaseDataset

from .trainer_config import TrainerConfig
from .lr_schedules import SCHEDULE_MAP

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

def split_batch( batch: BatchFeature, size: int ):
    items_list = {
        k: torch.split( v, size ) for k, v in batch.items()
    }

    lengths = len( items_list[ 'input_ids' ] )

    return [
        BatchFeature( { k: v[i] for k, v in items_list.items() } )
        for i in range( lengths )
    ]


@dataclass
class TrainingSchedule:
    samples_per_epoch: int
    total_training_steps: int
    validation_interval_steps: int
    evaluation_interval_steps: int

class Trainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
    ):
        mp.set_start_method( 'spawn' )
        
        self.trainer_config = trainer_config

        if trainer_config.stage == 'stage1':
            processor, model = self.load_pipeline_stage1()
        else:
            raise ValueError( 'stage2 not yet implemented!' )

        self.processor = processor
        self.model = model
        self.device = model.device

        self.dataset = self.get_dataset()
        self.training_schedule = self.get_training_schedule()
        self.optimizer = self.get_optimizer()
        self.lr_schedule = self.get_lr_schedule()

        self.accumulation_steps = self.trainer_config.batch_size // self.trainer_config.micro_batch_size
        self.train_step = 0

        self.train_forward_pass = torch.compile( self._train_forward_pass, mode=self.trainer_config.train_compile_mode ) if self.trainer_config.train_compile_mode is not None else self._train_forward_pass
        self.validation_forward_pass = torch.compile( self._validation_forward_pass, mode=self.trainer_config.validation_compile_mode ) if self.trainer_config.validation_compile_mode is not None else self._validation_forward_pass
        
    def load_pipeline_stage1( self ) -> tuple[ContextPeftProcessor, ContextPeftForConditionalGeneration]:
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

            attn_implementation='sdpa',
        )

        model = ContextPeftForConditionalGeneration( config, load_from_hub=True )
        model.train()

        if torch.cuda.is_available():
            model.cuda() # type: ignore

        return processor, model
        
    def get_dataset( self ) -> BaseDataset:
        if self.trainer_config.dataset == 'coco':
            dataset = CocoDataset(
                processor=self.processor,
                assistant_prefix='<|im_start|>assistant\n',
                assistant_suffix='<|im_end|>',
                batch_size=self.trainer_config.micro_batch_size,
                sequence_length=self.trainer_config.sequence_length,
                download_timeout=4 * 60 * 60,
            )
        else:
            raise ValueError( f'Invalid dataset {self.trainer_config.dataset}' )

        if self.trainer_config.sequence_length == -1:
            upad, pad = dataset.set_optimal_sequence_length( self.trainer_config.pad_to_multiple )
            print( f'Found max sequence length of {upad}, setting sequence length to {pad} due to rounding!' )

        return dataset

    def get_training_schedule( self ) -> TrainingSchedule:
        samples_per_epoch = len( self.dataset.get_train_split() )
        samples_total = samples_per_epoch * self.trainer_config.num_train_epochs
        batches_total = samples_total / self.trainer_config.batch_size
        log_steps_total = batches_total / self.trainer_config.logging_steps

        total_logs = math.ceil( log_steps_total )
        total_training_steps = total_logs * self.trainer_config.logging_steps
        validation_interval_steps = self.trainer_config.logging_steps * self.trainer_config.validation_interval
        evaluation_interval_steps = self.trainer_config.logging_steps * self.trainer_config.evaluation_interval

        return TrainingSchedule(
            samples_per_epoch=samples_per_epoch,
            total_training_steps=total_training_steps,
            validation_interval_steps=validation_interval_steps,
            evaluation_interval_steps=evaluation_interval_steps,
        )

    def get_optimizer( self ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            params=self.get_param_groups(),
            lr=0.0,
            betas=( self.trainer_config.adam_beta1, self.trainer_config.adam_beta2 ),
            weight_decay=0.0,
        )

        return optimizer

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

    def get_lr_schedule( self ):
        if self.trainer_config.warmup_steps < 1.0:
            warmup_steps = int( self.training_schedule.total_training_steps * self.trainer_config.warmup_steps )
        else:
            warmup_steps = int( self.trainer_config.warmup_steps )

        return SCHEDULE_MAP[self.trainer_config.learning_rate_schedule](
            warmup_steps=warmup_steps,
            total_training_steps=self.training_schedule.total_training_steps,
            lr=self.trainer_config.learning_rate,
            **self.trainer_config.learning_rate_schedule_kwargs,
        )

    def get_train_dataloader( self ):
        kwargs = {}

        # if self.device.type == 'cuda':
        #     kwargs[ 'pin_memory' ] = True
        #     kwargs[ 'pin_memory_device' ] = 'cuda'
        
        return self.dataset.train_dataloader(
            num_workers=self.trainer_config.dataset_train_workers,
            seed_start=hash( self.trainer_config.stage ),
            **kwargs
        )

    def get_validation_dataloader( self ):
        kwargs = {}

        # if self.device.type == 'cuda':
        #     kwargs[ 'pin_memory' ] = True
        #     kwargs[ 'pin_memory_device' ] = 'cuda'
        
        return self.dataset.validation_dataloader(
            worker=self.trainer_config.dataset_validation_worker,
            **kwargs
        )

    def _train_forward_pass( self, inputs: BatchFeature, labels: torch.Tensor ):
        with torch.autocast( self.device.type, dtype=torch.bfloat16 ):
            logits: torch.Tensor = self.model( **inputs, return_dict=True, use_cache=False ).logits

            B, S, D = logits.shape
            
            loss_mask = labels != -100

            acc = ( logits.argmax( -1 ) == labels ).float()
            acc = ( acc * loss_mask ).sum( -1 ) / loss_mask.sum( -1 )
            acc = acc.mean()

            loss = torch.nn.functional.cross_entropy(
                input=logits.reshape( B * S, D ).float(),
                target=labels.reshape( B * S ),
                reduction='none',
                ignore_index=-100
            ).reshape( B, S )

            loss = ( loss * loss_mask ).sum( -1 ) / loss_mask.sum( -1 )
            ppl = loss.exp()
            
            loss = loss.mean()
            ppl = ppl.mean()

        ( loss / self.accumulation_steps ).backward()
            
        return loss.detach(), acc.detach(), ppl.detach()

    def _validation_forward_pass( self, inputs: BatchFeature, labels: torch.Tensor ):
        with torch.no_grad():
            with torch.autocast( self.device.type, dtype=torch.bfloat16 ):
                logits: torch.Tensor = self.model( **inputs, return_dict=True, use_cache=False ).logits

                B, S, D = logits.shape
                
                loss_mask = labels != -100

                acc = ( logits.argmax( -1 ) == labels ).float()
                acc = ( acc * loss_mask ).sum( -1 ) / loss_mask.sum( -1 )
                acc = acc.mean()

                loss = torch.nn.functional.cross_entropy(
                    input=logits.reshape( B * S, D ).float(),
                    target=labels.reshape( B * S ),
                    reduction='none',
                    ignore_index=-100
                ).reshape( B, S )

                loss = ( loss * loss_mask ).sum( -1 ) / loss_mask.sum( -1 )
                ppl = loss.exp()
                
                loss = loss.mean()
                ppl = ppl.mean()
                
            return loss.detach(), acc.detach(), ppl.detach()

    def validation( self ):
        loss_metric = metrics.Mean().to( self.device )
        acc_metric = metrics.Mean().to( self.device )
        ppl_metric = metrics.Mean().to( self.device )

        self.model.eval()

        for micro_batch in tqdm.tqdm( self.get_validation_dataloader(), total=len( self.dataset.get_validation_split() ), smoothing=0.0, disable=True ):
            micro_batch = micro_batch.to( self.device )
            labels: torch.Tensor = micro_batch.pop( 'labels' )

            loss, acc, ppl = self.validation_forward_pass( micro_batch, labels )

            loss_metric.update( loss )
            acc_metric.update( acc )
            ppl_metric.update( ppl )

        loss = loss_metric.compute().item()
        acc = acc_metric.compute().item()
        ppl = ppl_metric.compute().item()

        return loss, acc, ppl

    def evaluation( self ):
        self.model.eval()

        for inputs, targets in tqdm.tqdm( self.dataset.evaluation_dataloader( True ), total=len( self.dataset.get_validation_split() ) ):
            inputs = inputs.to( self.device )

            input_len = inputs.input_ids.shape[-1]

            with torch.autocast( device_type=self.device.type, dtype=torch.bfloat16 ):
                out = self.model.generate(
                    **inputs,
                    pad_token_id=self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id,
                    max_length=self.dataset.sequence_length,
                    do_sample=False,
                    return_dict_in_generate=False,
                    skip_unused_adaptors=True
                )

            assert isinstance( out, torch.Tensor )
            print( *targets, sep='\n' )
            print( 'Prediction:', repr( self.processor.tokenizer.decode( out.squeeze( 0 ).cpu().tolist()[ input_len : ], skip_special_tokens=True ) ) )
        

    def train( self ):
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        train_iterator = iter( self.get_train_dataloader() )

        accumulation_steps = self.trainer_config.batch_size // self.trainer_config.micro_batch_size

        loss_metric = metrics.Mean().to( self.device )
        acc_metric = metrics.Mean().to( self.device )
        ppl_metric = metrics.Mean().to( self.device )

        for _ in tqdm.tqdm( range( self.training_schedule.total_training_steps ), smoothing=0.0, disable=True ):            
            self.model.train()
            for _ in range( accumulation_steps ):
                micro_batch: BatchFeature = next( train_iterator ).to( device=self.device )
                labels: torch.Tensor = micro_batch.pop( 'labels' )
                loss, acc, ppl = self.train_forward_pass( micro_batch, labels )

                loss_metric.update( loss )
                acc_metric.update( acc )
                ppl_metric.update( ppl )
                
                # print( f'{loss_metric.compute().item():.2f}, {acc_metric.compute().item():.2f}, {ppl_metric.compute().item():.2f}' )
                    
            self.train_step += 1

            for p_group in self.optimizer.param_groups:
                p_group[ 'lr' ] = self.lr_schedule.get_lr( self.train_step )

            torch.nn.utils.clip_grad_norm_( self.model.parameters(), self.trainer_config.max_grad_norm )
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.train_step % self.training_schedule.evaluation_interval_steps == 0 or self.train_step == self.training_schedule.total_training_steps:
                self.evaluation()
                print( f'Evaluation time {self.train_step}' )

            if self.train_step % self.training_schedule.validation_interval_steps == 0 or self.train_step == self.training_schedule.total_training_steps:
                loss, acc, ppl = self.validation()
                print( f'Validation time {self.train_step} | {loss:.2f} {acc:.2f} {ppl:.2f}' )

            if self.train_step % self.trainer_config.logging_steps == 0 or self.train_step == self.training_schedule.total_training_steps:
                percent = round(self.train_step*self.trainer_config.batch_size/self.training_schedule.samples_per_epoch*100,2)
                loss = loss_metric.compute().item()
                acc = acc_metric.compute().item()
                ppl = ppl_metric.compute().item()
                print( f'Logging time {self.train_step} | {percent}% | {loss:.2f} {acc:.2f} {ppl:.2f}' )
                loss_metric.reset()
                acc_metric.reset()
                ppl_metric.reset()

            if self.train_step == self.training_schedule.total_training_steps:
                print( 'Done!' )

            

            