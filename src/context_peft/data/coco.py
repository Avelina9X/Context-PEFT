from collections.abc import Iterator
from io import BytesIO
import os
import math
import random
import base64

import aiohttp

from PIL import Image
import tqdm
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    DownloadManager,
    DownloadConfig,
    are_progress_bars_disabled
)

from .base_dataset import BaseDataset
from .dataset_utils import (
    compute_assistant_mask,
    compute_f1,
    make_multimodal_assistant_turn,
    make_multimodal_user_turn,
    make_multimodal_user_turn_from_images,
)


def _compute_instruction_map() -> list[tuple[str, ...]]:
    CAPTION_INSTRUCTIONS_RAW = [
        # Standard caption
        'Describe this image: <image>',

        # Direct instructions
        'Describe this {IMAGE}:{SEP}<image>',
        'Write a caption for the {IMAGE}:{SEP}<image>',
        'Write a caption for the {IMAGE}:{SEP}<image>\nCaption:',
        'Provide a short description of what\'s shown in the {IMAGE}:{SEP}<image>',
        'Summarize the visual content of this {IMAGE}:{SEP}<image>',
        'Generate a natural language caption for this {IMAGE}:{SEP}<image>',

        # Instructional tone
        'As an AI assistant, write a descriptive caption for the {IMAGE}:{SEP}<image>',
        'You are tasked with generating a caption of {AN} {IMAGE}. Here\'s the {IMAGE}:{SEP}<image>',
        'Look at the {IMAGE} and return a one-sentence description:{SEP}<image>',
        'Write a caption that best describes the main scene in the {IMAGE}:{SEP}<image>',
        'Describe the scene in this {IMAGE} in plain English:{SEP}<image>',

        # Conversational style
        'What would be a good caption for this {IMAGE}?{SEP}<image>',
        'How would you describe this {IMAGE} in one sentence?{SEP}<image>',
        'If you were to describe this {IMAGE} to someone, what would you say?{SEP}<image>',

        # Direct
        '{IMAGE}:{SEP}<image>\nCaption:',
        'Caption this {IMAGE}:{SEP}<image>',
        'Caption this {IMAGE}:{SEP}<image>\nCaption:',
    ]

    IMAGE_INSTRUCTION_MAP = [ 'an image', 'a picture', 'a photo' ]
    SEP_INSTRUCTION_MAP = [ '', ' ', '\n' ]

    CAPTION_INSTRUCTIONS_FORMATTED = []

    for caption in CAPTION_INSTRUCTIONS_RAW:
        for sep_rep in SEP_INSTRUCTION_MAP:
            for an_image_rep in IMAGE_INSTRUCTION_MAP:
                an_rep, image_rep = an_image_rep.split( ' ' )

                text = caption.replace( '{IMAGE}', image_rep )
                text = text.replace( '{AN}', an_rep )
                text = text.replace( '{SEP}', sep_rep )

                text = text[0].upper() + text[1:]
                texts = text.split( '<image>' )
                texts.insert( 1, '<image>' )

                CAPTION_INSTRUCTIONS_FORMATTED.append( tuple( i for i in texts if i ) )
    
    return CAPTION_INSTRUCTIONS_FORMATTED

CAPTION_INSTRUCTIONS_MAP = _compute_instruction_map()


def _train_collate_fn(
    examples: list,
    coco_train_folder: str,
    padding_size: int,
    processor: ProcessorMixin,
    assistant_prefix: list[int],
    assistant_suffix: list[int],
    caption_instruction_map: list[tuple[str, ...]],
):
    pad_to = padding_size + 1

    messages = []

    for example in examples:
        prompt = random.choice( caption_instruction_map )
        caption = example[ 'caption' ]
        path = os.path.join( coco_train_folder, example[ 'file_name' ] )
        image = Image.open( path )
        image.load()

        messages.append( [
            make_multimodal_user_turn_from_images( prompt, [ image ] ),
            make_multimodal_assistant_turn( caption.strip() )
        ] )

    batch: BatchFeature = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors='pt',
        padding='max_length',
        max_length=pad_to,
        return_dict=True,
        truncation=False,
    ) # type: ignore

    input_ids, labels = compute_assistant_mask( batch.input_ids, assistant_prefix, assistant_suffix )

    inputs = {
        'input_ids': input_ids,
        'labels': labels,
    }

    if 'pixel_values' in batch and batch.pixel_values is not None:
        inputs[ 'pixel_values' ] = batch.pixel_values

    if 'attention_mask' in batch and batch.attention_mask is not None:
        inputs[ 'attention_mask' ] = batch.attention_mask[ :, : -1 ]

    return BatchFeature( inputs, tensor_type='pt' )

def _validation_collate_fn(
    examples: list,
    padding_size: int,
    processor: ProcessorMixin,
    assistant_prefix: list[int],
    assistant_suffix: list[int],
    caption_instruction: tuple[str, ...],
    valid_image_cache,
):
    pad_to = padding_size + 1

    messages = []

    for example in examples:
        image = valid_image_cache[ example[ 'file_name' ] ]
        for caption in example[ 'captions' ]:
            messages.append( [
                make_multimodal_user_turn_from_images( caption_instruction, [ image ] ),
                make_multimodal_assistant_turn( caption.strip() )
            ] )

    batch: BatchFeature = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors='pt',
        padding='max_length',
        max_length=pad_to,
        return_dict=True,
        truncation=False,
    ) # type: ignore

    input_ids, labels = compute_assistant_mask( batch.input_ids, assistant_prefix, assistant_suffix )

    inputs = {
        'input_ids': input_ids,
        'labels': labels,
    }

    if 'pixel_values' in batch and batch.pixel_values is not None:
        inputs[ 'pixel_values' ] = batch.pixel_values

    if 'attention_mask' in batch and batch.attention_mask is not None:
        inputs[ 'attention_mask' ] = batch.attention_mask[ :, : -1 ]

    return BatchFeature( inputs, tensor_type='pt' )

def _evaluation_collate_fn(
    examples: list[dict],
    processor: ProcessorMixin,
    caption_instruction: tuple[str, ...],
    valid_image_cache,
):
    message_list = []
    caption_list = []
    for example in examples:
        image = valid_image_cache[ example[ 'file_name' ] ]
        
        message = [ make_multimodal_user_turn_from_images( caption_instruction, [ image ] ) ]

        captions = [ caption.strip() for caption in example[ 'captions' ] ]

        message_list.append( message )
        caption_list.append( captions )
    
    batch: BatchFeature = processor.apply_chat_template(
        message_list,
        tokenize=True,
        return_tensors='pt',
        return_dict=True,
        add_generation_prompt=True,
    ) # type: ignore

    return batch, caption_list

def _split_captions_fn( examples ):
    rows = []
    for i in range( len( examples[ 'captions' ] ) ):
        for ids, caption in zip( examples[ 'ids' ][i], examples[ 'captions' ][i] ):
            rows.append( {
                **{ k: v[i] for k, v in examples.items() if k not in [ 'ids', 'captions' ] },
                'id': ids,
                'caption': caption,
            } )

    return { 'rows': rows }

def _split_captions( ds: Dataset ) -> Dataset:
    ds = ds.map( _split_captions_fn, batched=True, remove_columns=ds.column_names ).flatten()
    ds = ds.rename_columns( { k: k.split( '.' )[ -1 ] for k in ds.column_names } )
    return ds


class CocoDataset( BaseDataset ):
    """ COCO 2017 image captioning dataset. """

    def __init__(
        self,
        processor: ProcessorMixin,
        assistant_prefix: list[int] | str,
        assistant_suffix: list[int] | str,
        batch_size: int,
        sequence_length: int,
        cache_dir: str | None = None,
        download_timeout: int = 3600,
    ):
        """ Instantiates a COCO 2017 dataset.

        Args:
            processor (ProcessorMixin): Input processor with a tokenizer, image processor and chat template support.
            assistant_prefix (list[int] | str): Prefix of assistant messages. May be a string or list of token ids.
            assistant_suffix (list[int] | str): Suffix of assistant messages. May be a string or list of token ids.
            batch_size (int): Training batch size.
            sequence_length (int): Sequence length, will pad all sequences up to this size.
            cache_dir (str | None, optional): When specified uses this directory for dataset cache instead of default. Defaults to None.
        """
        super().__init__(
            processor=processor,
            assistant_prefix=assistant_prefix,
            assistant_suffix=assistant_suffix, 
            batch_size=batch_size,
            sequence_length=sequence_length
        )

        # Create download config with extended timeout, optional cache_dir, and optional progbars
        dl_config = DownloadConfig(
            cache_dir=cache_dir,
            storage_options={ 'client_kwargs': { 'timeout': aiohttp.ClientTimeout( total=download_timeout ) } },
            disable_tqdm=are_progress_bars_disabled(),
        )

        # Download training and validation images to a reusable location
        dl = DownloadManager( 'coco_2017_local', download_config=dl_config )
        train_folder = dl.download_and_extract( 'http://images.cocodataset.org/zips/train2017.zip' )
        valid_folder = dl.download_and_extract( 'http://images.cocodataset.org/zips/val2017.zip' )

        assert isinstance( train_folder, str )
        assert isinstance( valid_folder, str )

        self.train_folder = train_folder
        self.valid_folder = valid_folder

        # Download the captions dataset
        dataset = load_dataset( 'phiyodr/coco2017', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )

        self.valid_image_cache = {}
        for row in tqdm.tqdm( dataset[ 'validation' ], desc='Loading validation images' ):
            file_name = row[ 'file_name' ]
            file_path = os.path.join( valid_folder, file_name )
            image = Image.open( file_path )
            image.load()
            self.valid_image_cache[ file_name ] = image

        # Apply map to split captions
        self.train_split = _split_captions( dataset[ 'train' ] )

        # Set validation split as is
        self.valid_split = dataset[ 'validation' ]

    def get_name( self ) -> str:
        return 'coco'
    
    def get_train_split( self ) -> Dataset:
        return self.train_split
    
    def get_validation_split( self ) -> Dataset:
        return self.valid_split
    
    def train_collate_fn( self, examples: list ) -> BatchFeature:
        return _train_collate_fn(
            examples=examples,
            coco_train_folder=self.train_folder,
            padding_size=self.sequence_length,
            processor=self.processor,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
            caption_instruction_map=CAPTION_INSTRUCTIONS_MAP,
        )
    
    def validation_collate_fn( self, examples: list ) -> BatchFeature:
        return _validation_collate_fn(
            examples=examples,
            padding_size=self.sequence_length,
            processor=self.processor,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
            caption_instruction=CAPTION_INSTRUCTIONS_MAP[0],
            valid_image_cache=self.valid_image_cache
        )

    def evaluation_collate_fn( self, examples: list[dict] ) -> tuple[BatchFeature, list[list[str]]]:
        return _evaluation_collate_fn(
            examples=examples,
            processor=self.processor,
            caption_instruction=CAPTION_INSTRUCTIONS_MAP[0],
            valid_image_cache=self.valid_image_cache
        )
    
    def validation_iterator( self ) -> Iterator[BatchFeature]:
        for row in self.get_validation_split():
            yield self.validation_collate_fn( [ row ] )

    def evaluation_iterator( self, batch_size: int | None = None ) -> Iterator[tuple[BatchFeature, list[list[str]]]]:
        rows = []
        for row in self.get_validation_split():
            assert isinstance( row, dict )

            rows.append( row )

            if len( rows ) == ( batch_size or self.batch_size ):
                yield self.evaluation_collate_fn( rows )
                rows = []
        if rows:
            yield self.evaluation_collate_fn( rows )

    def compute_score( self, prediction: str, references: list[str], **kwargs ) -> dict[str, float]:
        f1, precision, recall = compute_f1( prediction, references )
        return { 'f1': f1, 'precision': precision, 'recall': recall }

    def set_optimal_sequence_length( self, pad_to_multiple=32, image_seq_len: int | None = None ) -> tuple[int, int]:
        # Get the longest prompt (in image placeholder mode)
        longest_prompt: tuple[str, ...] = ()
        longest_prompt_len = 0
        for prompt in CAPTION_INSTRUCTIONS_MAP:
            messages = [ make_multimodal_user_turn( prompt, [ '' ] ) ]
            prompt_str = self.processor.apply_chat_template( messages )
            prompt_len = len( self.tokenizer.encode( prompt_str ) )

            if prompt_len > longest_prompt_len:
                longest_prompt = prompt
                longest_prompt_len = prompt_len


        # Get the longest sequence with the longest prompt (in image placeholder mode)
        longest_example_len = 0
        for example in tqdm.tqdm( self.get_train_split(), desc='Computing train sequence length' ):
            assert isinstance( example, dict )
            messages = [
                make_multimodal_user_turn( longest_prompt, [ '' ] ),
                make_multimodal_assistant_turn( example[ 'caption' ].strip() )
            ]
            example_str = self.processor.apply_chat_template( messages )
            example_len = len( self.tokenizer.encode( example_str ) )

            if example_len > longest_example_len:
                longest_example_len = example_len

        for example in tqdm.tqdm( self.get_validation_split(), desc='Computing validation sequence length' ):
            assert isinstance( example, dict )
            for caption in example[ 'captions' ]:
                messages = [
                    make_multimodal_user_turn( CAPTION_INSTRUCTIONS_MAP[0], [ '' ] ),
                    make_multimodal_assistant_turn( caption.strip() )
                ]
                example_str = self.processor.apply_chat_template( messages )
                example_len = len( self.tokenizer.encode( example_str ) )

                if example_len > longest_example_len:
                    longest_example_len = example_len

        
        # Get the image sequence length if not set
        image_seq_len = image_seq_len or getattr( self.processor, 'image_seq_len', None )
        image_seq_len = image_seq_len or getattr( self.processor, 'image_seq_length', None )

        if image_seq_len is None:
            raise ValueError( 'image_seq_len could not be inferred from processor. Please pass image_seq_len!' )

        # Because length contains a placeholder token we subtract 1 and then add image_seq_len and 2 for BOS/EOS
        unpadded_sequence_length = longest_example_len - 1 + image_seq_len + 2

        # Round up to next multiple
        padded_sequence_length = math.ceil( unpadded_sequence_length / pad_to_multiple ) * pad_to_multiple

        # Finally, set the sequence length
        self.sequence_length = padded_sequence_length

        return unpadded_sequence_length, padded_sequence_length