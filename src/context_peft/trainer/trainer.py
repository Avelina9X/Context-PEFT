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

import safetensors.torch
from transformers import CLIPVisionModel, AutoImageProcessor, AutoTokenizer, AutoConfig, BatchFeature
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.cache_utils import StaticCache

from model import ContextPeftConfig, ContextPeftProcessor, ContextPeftForConditionalGeneration
from model.modeling_context_peft import CONTEXT_PEFT_WRAPPER_MAPPING

from data import BaseDataset, CocoDataset, LlavaInstructDataset, GQADataset, VQADataset

from .trainer_config import TrainerConfig
from .lr_schedules import SCHEDULE_MAP

from .trainer_utils import seed_hash, get_adaptors, get_peft_config

@dataclass
class TrainingSchedule:
    samples_per_epoch: int
    total_training_steps: int
    validation_interval_steps: int
    evaluation_interval_steps: int

class Trainer:
    def __init__( self, trainer_config: TrainerConfig ):
        self.trainer_config = trainer_config
        
        mp.set_start_method( 'spawn' )

        torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
        torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
        torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 # type: ignore # pylint: disable=W0212
        torch._dynamo.config.compiled_autograd = False # type: ignore # pylint: disable=W0212

        if trainer_config.wandb_mode == 'disabled' or True:
            torch._logging.set_logs(
                graph_breaks=True,
                recompiles=True,
            )

        if trainer_config.stage == 'stage1':
            processor, model, dirty_parameters = self.load_pipeline_stage1()
        elif trainer_config.stage == 'stage2':
            processor, model, dirty_parameters = self.load_pipeline_stage2()
        else:
            raise ValueError( f'Stage {trainer_config.stage} not yet implemented!' )
        

        self.processor = processor
        self.model = model
        self.device = model.device
        self.dirty_parameters = dirty_parameters

        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.dataset = self.get_dataset( self.trainer_config.dataset )
        self.evaluation_datasets = [ self.get_dataset( d ) for d in self.trainer_config.evaluation_datasets ]
        
        self.training_schedule = self.get_training_schedule()
        self.optimizer = self.get_optimizer()
        self.lr_schedule = self.get_lr_schedule()

        self.accumulation_steps = trainer_config.batch_size // trainer_config.micro_batch_size
        self.training_step = 0

        self.train_forward_pass = (
            torch.compile( self._train_forward_pass, mode=trainer_config.train_compile_mode, dynamic=False )
            if trainer_config.train_compile_mode is not None
            else self._train_forward_pass
        )
        self.validation_forward_pass = None

        self._train_iterator = self.get_train_dataloader()
        self._validation_iterator = self.get_validation_dataloader()
        self._evaluation_iterator = self.get_evaluation_dataloader( self.dataset )
        self._evaluation_caches: dict[int, StaticCache] = {}

        self.loss_metric = metrics.Mean().to( self.device )
        self.acc_metric = metrics.Mean().to( self.device )
        self.ppl_metric = metrics.Mean().to( self.device )

        self.train_samplerate_metric = metrics.Mean()
        self.total_train_metric = metrics.Sum()

        self.grad_norm_max = metrics.Max().to( self.device )
        self.grad_norm_avg = metrics.Mean().to( self.device )
        
    def load_pipeline_stage1( self ) -> tuple[ContextPeftProcessor, ContextPeftForConditionalGeneration, set[str]]:
        vision_model_name = self.trainer_config.vision_model_name
        text_model_name = self.trainer_config.text_model_name
        assert vision_model_name is not None
        assert text_model_name is not None
        
        vision_config = CLIPVisionModel.from_pretrained( vision_model_name ).config
        vision_processor = AutoImageProcessor.from_pretrained( vision_model_name, use_fast=True )
        image_seq_len = ( vision_config.image_size // vision_config.patch_size ) ** 2 # TODO: set via config with CLS # FIXME

        text_config = AutoConfig.from_pretrained( text_model_name )
        text_tokenizer = AutoTokenizer.from_pretrained( text_model_name, eos_token='<|im_end|>', use_fast=True )

        processor = ContextPeftProcessor(
            image_processor=vision_processor,
            tokenizer=text_tokenizer,
            image_seq_len=image_seq_len,
            chat_template='chat_ml'
        )

        peft_config = get_peft_config(
            self.trainer_config.peft_type,
            self.trainer_config.adaptor_kwargs
        )

        adaptors = get_adaptors(
            self.trainer_config.dataset,
            self.trainer_config.adaptor_context,
            text_config.num_hidden_layers,
            self.trainer_config.peft_type,
            self.trainer_config.lora_image_scale
        )

        config = ContextPeftConfig(
            vision_config=vision_config,
            vision_dim=vision_config.hidden_size,
            vision_trainable=False,

            text_config=text_config,
            text_dim=text_config.hidden_size,
            text_trainable=self.trainer_config.text_trainable,

            image_pad_token_id=processor.get_image_pad_token_id(),

            connector_trainable=True,
            connector_dropout=self.trainer_config.connector_dropout,
            connector_bias=self.trainer_config.connector_bias,

            peft_type=self.trainer_config.peft_type,
            default_peft_config=peft_config,
            adaptors=adaptors,

            attn_implementation='sdpa',

            bos_token_id=text_tokenizer.bos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            eos_token_id=text_tokenizer.eos_token_id,
            sep_token_id=text_tokenizer.sep_token_id,
        )

        model = ContextPeftForConditionalGeneration(
            config,
            load_from_hub=True,
            device_map='cuda' if torch.cuda.is_available() else 'cpu'
        )

        if config.text_trainable: # pylint: disable=E1101
            model.text_model.float()

        if config.vision_trainable: # pylint: disable=E1101
            model.vision_model.float()

        if config.connector_trainable:
            model.connector.float()

        if self.trainer_config.trainable_embeddings:
            model.text_model.get_input_embeddings().requires_grad_( True )
            model.text_model.get_output_embeddings().requires_grad_( True )
            model.text_model.get_input_embeddings().float()
            model.text_model.get_output_embeddings().float()
        else:
            model.text_model.get_input_embeddings().requires_grad_( False )
            model.text_model.get_output_embeddings().requires_grad_( False )
            model.text_model.get_input_embeddings().bfloat16()
            model.text_model.get_output_embeddings().bfloat16()
        
        model.train()

        dirty_parameters: set[str] = set()
        for n, p in model.named_parameters():
            if p.requires_grad:
                dirty_parameters.add( n )

        if torch.cuda.is_available():
            model.cuda() # type: ignore
            torch.backends.cuda.enable_math_sdp( False )

        return processor, model, dirty_parameters

    def load_pipeline_stage2( self ) -> tuple[ContextPeftProcessor, ContextPeftForConditionalGeneration, set[str]]:
        cpeft_model_path = self.trainer_config.cpeft_model_path
        assert cpeft_model_path is not None

        cpeft_model_path = cpeft_model_path.format( **os.environ )

        processor = ContextPeftProcessor.from_pretrained( cpeft_model_path )
        assert isinstance( processor, ContextPeftProcessor )

        text_config = ContextPeftConfig.from_pretrained( cpeft_model_path ).get_text_config()

        peft_config = get_peft_config(
            self.trainer_config.peft_type,
            self.trainer_config.adaptor_kwargs,
        )

        adaptors = get_adaptors(
            self.trainer_config.dataset,
            self.trainer_config.adaptor_context,
            text_config.num_hidden_layers,
            self.trainer_config.peft_type,
            self.trainer_config.lora_image_scale,
        )

        config = ContextPeftConfig.from_pretrained(
            cpeft_model_path,

            text_trainable=self.trainer_config.text_trainable,

            connector_trainable=True,
            connector_dropout=self.trainer_config.connector_dropout,
            connector_bias=self.trainer_config.connector_bias,

            peft_type=self.trainer_config.peft_type,
            default_peft_config=peft_config,
            adaptors=adaptors,

            attn_implementation='sdpa',
        )

        assert isinstance( config, ContextPeftConfig )

        model = ContextPeftForConditionalGeneration(
            config=config,
            load_from_hub=True,
            device_map='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        adaptor_path = os.path.join( cpeft_model_path, 'adaptors.safetensors' )
        state_dict = safetensors.torch.load_file(
            adaptor_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        dirty_parameters: set[str] = set( state_dict.keys() )

        incompatible_keys = model.load_state_dict( state_dict, strict=False )
        assert( len( incompatible_keys.unexpected_keys ) == 0 )

        if config.text_trainable: # pylint: disable=E1101
            model.text_model.float()

        if config.vision_trainable: # pylint: disable=E1101
            model.vision_model.float()

        if config.connector_trainable: # pylint: disable=E1101
            model.connector.float()

        if self.trainer_config.trainable_embeddings:
            model.text_model.get_input_embeddings().requires_grad_( True )
            model.text_model.get_output_embeddings().requires_grad_( True )
            model.text_model.get_input_embeddings().float()
            model.text_model.get_output_embeddings().float()
        else:
            model.text_model.get_input_embeddings().requires_grad_( False )
            model.text_model.get_output_embeddings().requires_grad_( False )
            model.text_model.get_input_embeddings().bfloat16()
            model.text_model.get_output_embeddings().bfloat16()

        model.train()

        for n, p in model.named_parameters():
            if p.requires_grad:
                dirty_parameters.add( n )

        if torch.cuda.is_available():
            model.cuda() # type: ignore
            torch.backends.cuda.enable_math_sdp( False )

        return processor, model, dirty_parameters
        
        
    def get_dataset( self, dataset_name: str ) -> BaseDataset:
        args = {
            'processor': self.processor,
            'assistant_prefix': '<|im_start|>assistant\n',
            'assistant_suffix': '<|im_end|>',
            'batch_size': self.trainer_config.micro_batch_size,
            'sequence_length': self.trainer_config.sequence_length,
        }
        
        if dataset_name == 'coco':
            dataset = CocoDataset( **args, download_timeout=4 * 60 * 60 )
        elif dataset_name == 'llava150k':
            dataset = LlavaInstructDataset( **args, download_timeout=4 * 60 * 60 )
        elif dataset_name == 'gqa':
            dataset = GQADataset( **args )
        elif dataset_name == 'vqav2':
            dataset = VQADataset( **args, download_timeout=4 * 60 * 60 )
        else:
            raise ValueError( f'Invalid dataset {dataset_name}' )

        assert dataset_name == dataset.get_name()

        if self.trainer_config.sequence_length == -1:
            upad, pad = dataset.set_optimal_sequence_length( self.trainer_config.pad_to_multiple )
            print( f'Found max sequence length of {upad}, setting sequence length to {pad} due to rounding!' )

            self.trainer_config.sequence_length = pad

        return dataset

    def get_training_schedule( self ) -> TrainingSchedule:
        samples_per_epoch = len( self.dataset.get_train_split() )
        samples_total = samples_per_epoch * self.trainer_config.num_train_epochs
        batches_total = samples_total / self.trainer_config.batch_size
        log_steps_total = batches_total / self.trainer_config.logging_steps

        total_logs = math.floor( log_steps_total )
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

        named_parameters = list( self.model.named_parameters() )

        for name, p in named_parameters:
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
                'params': [ p for n, p in named_parameters if n in base_decay_parameters ],
                'weight_decay': base_decay,
            },
            {
                'params': [ p for n, p in named_parameters if n in base_nocay_parameters ],
                'weight_decay': 0,
            },
            {
                'params': [ p for n, p in named_parameters if n in adaptor_parameters ],
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

        if self.trainer_config.dataset_train_workers > 0:
            kwargs[ 'prefetch_factor' ] = 4 * self.accumulation_steps

        if self.device.type == 'cuda':
            kwargs[ 'pin_memory' ] = True
            kwargs[ 'pin_memory_device' ] = 'cuda'
        
        return self.dataset.train_dataloader(
            num_workers=self.trainer_config.dataset_train_workers,
            seed_start=seed_hash( self.trainer_config.stage, self.trainer_config.seed_offset ),
            **kwargs
        )

    def get_validation_dataloader( self ):
        kwargs = {}

        if self.device.type == 'cuda':
            kwargs[ 'pin_memory' ] = True
            kwargs[ 'pin_memory_device' ] = 'cuda'

        if self.trainer_config.dataset_validation_worker:
            kwargs[ 'prefetch_factor' ] = 4
            kwargs[ 'persistent_workers' ] = True
        
        return self.dataset.validation_dataloader(
            worker=self.trainer_config.dataset_validation_worker,
            **kwargs
        )

    def get_evaluation_dataloader( self, dataset: BaseDataset ):
        kwargs = {}

        if self.device.type == 'cuda':
            kwargs[ 'pin_memory' ] = True
            kwargs[ 'pin_memory_device' ] = 'cuda'

        if self.trainer_config.dataset_validation_worker:
            kwargs[ 'prefetch_factor' ] = 4
            kwargs[ 'persistent_workers' ] = True
        
        return dataset.evaluation_dataloader(
            worker=self.trainer_config.dataset_validation_worker,
            **kwargs
        )


    def _validation_forward_pass( self, inputs: dict, labels: torch.Tensor ):
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
            
        return loss, acc, ppl

    @torch.no_grad
    def validation( self ):
        if self.validation_forward_pass is None:
            self.validation_forward_pass = (
                torch.compile( self._validation_forward_pass, mode=self.trainer_config.validation_compile_mode, fullgraph=True )
                if self.trainer_config.validation_compile_mode is not None
                else self._validation_forward_pass
            )
            
        loss_metric = metrics.Mean().to( self.device )
        acc_metric = metrics.Mean().to( self.device )
        ppl_metric = metrics.Mean().to( self.device )

        self.model.eval()

        iterator = iter( self._validation_iterator )
        length = len( self.dataset.get_validation_split() )

        start_time = time.time()

        for micro_batch in tqdm.tqdm( iterator, total=length, smoothing=0.0, ncols=60, disable=self.trainer_config.wandb_mode == 'online' ):
            micro_batch = micro_batch.to( self.device, non_blocking=True )
            labels: torch.Tensor = micro_batch.pop( 'labels' )
            micro_batch.pop( 'attention_mask' )

            loss, acc, ppl = self.validation_forward_pass( dict( **micro_batch ), labels )

            loss_metric.update( loss )
            acc_metric.update( acc )
            ppl_metric.update( ppl )

        loss = loss_metric.compute().item()
        acc = acc_metric.compute().item()
        ppl = ppl_metric.compute().item()

        end_time = time.time()

        metric_dict = {
            f'validation/{self.dataset.get_name()}/loss': loss,
            f'validation/{self.dataset.get_name()}/acc': acc,
            f'validation/{self.dataset.get_name()}/ppl': ppl,
            'stats/val_time': end_time - start_time,
        }

        return metric_dict


    def _evaluation_forward_pass( self, dataset: BaseDataset, iterator, log_table=False ):
        pred_batch = []
        targets_batch = []
        batch_sizes = []

        pad_token_id = self.processor.tokenizer.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id

        for inputs, targets in tqdm.tqdm( iterator, smoothing=0.0, ncols=60, disable=self.trainer_config.wandb_mode == 'online' ):
            inputs = inputs.to( self.device, non_blocking=True )

            batch_size, input_len = inputs.input_ids.shape

            assert batch_size == len( targets )

            if batch_size not in self._evaluation_caches:
                self._evaluation_caches[batch_size] = StaticCache(
                    self.model.text_model.config,
                    max_cache_len=dataset.sequence_length,
                    device=self.model.device,
                    dtype=torch.bfloat16,
                    max_batch_size=batch_size,
                )

            past_key_values = self._evaluation_caches[batch_size]
            past_key_values.reset()

            with torch.autocast( device_type=self.device.type, dtype=torch.bfloat16 ):
                out = self.model.generate(
                    **inputs,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    max_length=dataset.sequence_length,
                    do_sample=False,
                    return_dict_in_generate=False,
                    skip_unused_adaptors=False,
                    past_key_values=past_key_values,
                )

            assert isinstance( out, torch.Tensor )

            pred = out[ :, input_len : ]

            pred_batch.append( pred )
            targets_batch.append( targets )
            batch_sizes.append( batch_size )

        table_rows: list[tuple[str,list[str]]] = []
        
        for pred, targets, batch_size in zip( pred_batch, targets_batch, batch_sizes ):
            pred_cpu = pred.cpu().tolist()
            assert len( pred_cpu ) == len( targets )
            for i in range( batch_size ):
                pred_str: str = self.processor.tokenizer.decode( pred_cpu[i], skip_special_tokens=True )
                targets_list: list[str] = targets[i]
                table_rows.append( ( pred_str, targets_list ) )

        raw_scores = dataset.compute_scores(
            [ x[0] for x in table_rows ],
            [ x[1] for x in table_rows ]
        )
        
        metric_dict: dict = { f'evaluation/{dataset.get_name()}/{k}': v for k, v in raw_scores.items() }

        if log_table:
            table = wandb.Table( columns=[ 'pred', 'targets' ], data=table_rows )
            metric_dict[ f'predictions/{self.dataset.get_name()}' ] = table

        return metric_dict

    @torch.no_grad
    def evaluation( self, final=False ):
        self.model.eval()
        iterator = iter( self._evaluation_iterator )

        start_time = time.time()
        metric_dict = self._evaluation_forward_pass( self.dataset, iterator, log_table=final )
        end_time = time.time()

        metric_dict[ 'stats/eval_time' ] = end_time - start_time
        
        if final:
            for dataset in self.evaluation_datasets:
                iterator = iter( self.get_evaluation_dataloader( dataset ) )
                metric_dict.update(
                    **self._evaluation_forward_pass( dataset, iterator, log_table=False )
                )

        return metric_dict


    def _train_forward_pass( self, inputs: dict, labels: torch.Tensor ):
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
            ppl = loss.detach().exp()
            
            loss = loss.mean()
            ppl = ppl.mean()

        ( loss / self.accumulation_steps ).backward()
            
        return loss.detach(), acc.detach(), ppl.detach()

    def _train_step( self, train_iterator ):
        self.model.train()
        
        step_start_time = time.time()

        for _ in range( self.accumulation_steps ):
            micro_batch: BatchFeature = next( train_iterator ).to( device=self.device, non_blocking=True )
            labels: torch.Tensor = micro_batch.pop( 'labels' )
            micro_batch.pop( 'attention_mask' )
            loss, acc, ppl = self.train_forward_pass( dict( **micro_batch ), labels )

            self.loss_metric.update( loss )
            self.acc_metric.update( acc )
            self.ppl_metric.update( ppl )
                
        self.training_step += 1

        for p_group in self.optimizer.param_groups:
            p_group[ 'lr' ] = self.lr_schedule.get_lr( self.training_step )

        total_grad = torch.nn.utils.clip_grad_norm_( self.model.parameters(), self.trainer_config.max_grad_norm )
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.grad_norm_max.update( total_grad )
        self.grad_norm_avg.update( total_grad )

        step_end_time = time.time()

        time_delta = step_end_time - step_start_time
        samplerate = self.trainer_config.batch_size / time_delta

        self.train_samplerate_metric.update( torch.tensor( samplerate, dtype=torch.float ) )
        self.total_train_metric.update( torch.tensor( time_delta, dtype=torch.float ) )

    def train( self ):
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        train_iterator = iter( self._train_iterator )

        run = self.init_wandb()

        for _ in tqdm.tqdm(
            range( self.training_schedule.total_training_steps ),
            smoothing=0.0,
            ncols=60,
            disable=self.trainer_config.wandb_mode == 'online'
        ):            
            metric_dict = {}

            self._train_step( train_iterator )

            final_step = self.training_step == self.training_schedule.total_training_steps

            if self.training_step % self.training_schedule.evaluation_interval_steps == 0 or final_step:
                metric_dict.update( self.evaluation( final_step ) )

            if self.training_step % self.training_schedule.validation_interval_steps == 0 or final_step:
                metric_dict.update( self.validation() )

            if self.training_step % self.trainer_config.logging_steps == 0 or final_step:
                metric_dict.update( self.get_train_metric_dict() )
                metric_dict.update( self.get_stats_metric_dict() )
                metric_dict.update( self.get_grads_metric_dict() )
                
                print( self.get_log_string( metric_dict ), flush=True )

                run.log( metric_dict )

            if final_step:
                print( 'Done!' )

                if self.trainer_config.output_dir is not None:
                    output_dir = self.trainer_config.output_dir.format( **os.environ )
                    output_path = os.path.join( output_dir, self.trainer_config.run_name )

                    config_path = os.path.join( output_path, 'trainer_config.yaml' )

                    os.makedirs( output_path, exist_ok=True )

                    adaptor_path = os.path.join( output_path, 'adaptors.safetensors' )
                    safetensors.torch.save_file( self.get_params_to_save(), adaptor_path )
                        
                    self.processor.save_pretrained( output_path )

                    config_dict = dataclasses.asdict( self.trainer_config )

                    with open( config_path, 'w', encoding='utf-8' ) as f:
                        yaml.dump( config_dict, f, default_flow_style=False, sort_keys=False )
                    
                run.finish()
                break

    def init_wandb( self ):
        wandb.login( key=os.environ[ 'WANDB_API_KEY' ] )

        total_params = self.model.num_parameters( only_trainable=False )
        trainable_params = self.model.num_parameters( only_trainable=True )

        run = wandb.init(
            project=os.environ[ 'WANDB_PROJECT_NAME' ],
            mode=self.trainer_config.wandb_mode,
            name=self.trainer_config.run_name,
            group=self.trainer_config.wandb_group,
            tags=self.trainer_config.wandb_tags,
            config={
                'trainer_config': dataclasses.asdict( self.trainer_config ),
                'model_config': self.model.config.to_dict(),
                'params': {
                    'total': total_params,
                    'trainable': trainable_params,
                }
            }
        )

        return run


    def get_train_metric_dict( self ):
        metric_dict = {
            f'train/{self.dataset.get_name()}/loss': self.loss_metric.compute().item(),
            f'train/{self.dataset.get_name()}/acc': self.acc_metric.compute().item(),
            f'train/{self.dataset.get_name()}/ppl': self.ppl_metric.compute().item(),
        }

        self.loss_metric.reset()
        self.acc_metric.reset()
        self.ppl_metric.reset()

        return metric_dict

    def get_stats_metric_dict( self ):
        metric_dict = {
            'stats/train_step': self.training_step,
            'stats/dataset_epoch': self.training_step * self.trainer_config.batch_size / self.training_schedule.samples_per_epoch,
            'stats/learning_rate': self.lr_schedule.get_lr( self.training_step ),
            'stats/samplerate': self.train_samplerate_metric.compute().item(), # do NOT reset
            'stats/train_time': self.total_train_metric.compute().item(), # do NOT reset
        }

        return metric_dict

    def get_grads_metric_dict( self ):
        metric_dict = {
            'grads/max_norm': self.grad_norm_max.compute().item(),
            'grads/avg_norm': self.grad_norm_avg.compute().item(),
        }

        self.grad_norm_max.reset()
        self.grad_norm_avg.reset()

        return metric_dict

    def get_log_string( self, metric_dict: dict[str, Any] ):
        step = metric_dict[ 'stats/train_step' ]
        epoch = metric_dict[ 'stats/dataset_epoch' ]

        train_stats = {
            't_' + k.split( '/' )[-1]: v for k, v in metric_dict.items() if k.startswith( 'train/' )
        }

        valid_stats = {
            'v_' + k.split( '/' )[-1]: v for k, v in metric_dict.items() if k.startswith( 'validation/' )
        }

        eval_stats = {
            'e_' + k.split( '/' )[-1]: v for k, v in metric_dict.items() if k.startswith( 'evaluation/' )
        }

        all_metrics_dict = { **train_stats, **valid_stats, **eval_stats }
        all_metrics_list = [ f'{k}={v:.3f}' for k, v in all_metrics_dict.items() ]
        all_metrics_string = ' '.join( all_metrics_list )

        return f'step={step} | {epoch:.2f} | {all_metrics_string}'
            

    def get_params_to_save( self ):
        return { n: p for n, p in self.model.state_dict().items() if n in self.dirty_parameters }
