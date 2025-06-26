import os
import argparse
import yaml
import jsonlines

import tqdm
import torch

from data import BaseDataset, CocoDataset, compute_f1
from model.modeling_context_peft import ContextPeftForConditionalGeneration
from model.processing_context_peft import ContextPeftProcessor
from trainer import TrainerConfig

def get_dataset( processor, dataset_name: str, sequence_length: int ) -> BaseDataset:
    if dataset_name == 'coco':
        dataset = CocoDataset(
            processor=processor,
            assistant_prefix='<|im_start|>assistant\n',
            assistant_suffix='<|im_end|>',
            batch_size=1,
            sequence_length=sequence_length,
            download_timeout=4 * 60 * 60,
        )
    else:
        raise ValueError( f'Invalid dataset {dataset_name}' )

    if sequence_length == -1:
        upad, pad = dataset.set_optimal_sequence_length( 1 )
        print( f'Found max sequence length of {upad}, setting sequence length to {pad} due to rounding!' )

    return dataset

def get_evaluation_dataloader( dataset: BaseDataset ):
    kwargs = {}

    kwargs[ 'prefetch_factor' ] = 4

    if torch.cuda.is_available() == 'cuda':
        kwargs[ 'pin_memory' ] = True
        kwargs[ 'pin_memory_device' ] = 'cuda'
    
    return dataset.evaluation_dataloader(
        worker=True,
        **kwargs
    )

@torch.no_grad
def generate( model, processor, dataset ):
    iterator = iter( get_evaluation_dataloader( dataset ) )

    pred_batch = []
    targets_batch = []
    batch_sizes = []

    pred_list = []
    targets_list = []

    pad_token_id = processor.tokenizer.pad_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    length = len( dataset.get_validation_split() )

    for inputs, targets in tqdm.tqdm( iterator, smoothing=0.0, ncols=80, total=length ):
        inputs = inputs.to( model.device, non_blocking=True )
        
        batch_size, input_len = inputs.input_ids.shape

        assert batch_size == len( targets )

        with torch.autocast( device_type=model.device.type, dtype=torch.bfloat16 ):
            out = model.generate(
                **inputs,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=dataset.sequence_length,
                do_sample=False,
                return_dict_in_generate=False,
                skip_unused_adaptors=True
            )

        assert isinstance( out, torch.Tensor )

        pred = out[ :, input_len : ]

        pred_batch.append( pred )
        targets_batch.append( targets )
        batch_sizes.append( batch_size )

    for pred, targets, batch_size in zip( pred_batch, targets_batch, batch_sizes ):
        pred_cpu = pred.cpu().tolist()
        assert len( pred_cpu ) == len( targets )
        for i in range( batch_size ):
            pred_list.append( pred_cpu[i] )
            targets_list.append( targets[i] )

    output_list = []

    for pred_tokens, targets in zip( pred_list, targets_list ):
        pred = processor.tokenizer.decode( pred_tokens, skip_special_tokens=True )

        output_list.append( {
            'prediction': pred,
            'targets': targets,
        } )

    return output_list

def predict_from_path( model_path ):
    processor = ContextPeftProcessor.from_pretrained( model_path )
    model = ContextPeftForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype='auto',
        device_map='cuda' if torch.cuda.is_available() else 'cpu'
    )

    model.eval()

    config_path = os.path.join( model_path, 'trainer_config.yaml' )
    with open( config_path, 'r', encoding='utf-8' ) as f:
        config = yaml.safe_load( f )
        assert isinstance( config, dict )

    sequence_length = config[ 'sequence_length' ]
    dataset_name = config[ 'dataset' ]
    model_name = config[ 'run_name' ]

    dataset = get_dataset( processor, dataset_name, sequence_length )

    predictions = generate( model, processor, dataset )

    output_dir = os.path.join( model_path, 'predictions' )
    output_path = os.path.join( output_dir, f'{model_name}.{dataset_name}.jsonl' )

    os.makedirs( output_dir, exist_ok=True )

    with open( output_path, 'w' ) as f:
        jsonlines.Writer( f ).write_all( predictions )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model', type=str )
    parser.add_argument( '--config', type=str )
    args = parser.parse_args()

    if args.model is not None:
        model_path = args.model
    else:
        with open( args.config, 'r', encoding='utf-8' ) as f:
            config = yaml.safe_load( f )
            assert isinstance( config, dict )

        output_dir = config[ 'output_dir' ].format( **os.environ )
        model_name = config[ 'run_name' ]

        model_path = os.path.join( output_dir, model_name )

    print( f'path="{model_path}"')

    predict_from_path( model_path )

if __name__ == '__main__':
    main()