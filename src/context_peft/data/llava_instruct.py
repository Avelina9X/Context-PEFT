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
    compute_f1,
    messages_as_batch,
    make_multimodal_assistant_turn,
    make_multimodal_user_turn,
    make_multimodal_user_turn_from_images,
)

LLAVA_INSTRUCT_URL = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json'

def _train_conversation_mapper( example: dict, image ):
    conversations = []
    for message in example[ 'conversations' ]:
        role = { 'human': 'user', 'gpt': 'assistant' }[ message[ 'from' ] ]
        content = []
        for string in message[ 'value' ].partition( '<image>' ):
            if string == '<image>':
                content.append( { 'type': 'image', 'image': image } )
            elif string:
                content.append( { 'type': 'text', 'text': string } )
        conversations.append( { 'role': role, 'content': content } )
    return conversations

def _train_collate_fn(
    examples: list,
    coco_train_folder: str,
    padding_size: int,
    processor: ProcessorMixin,
    assistant_prefix: list[int],
    assistant_suffix: list[int],
):
    pad_to = padding_size + 1

    messages = []

    for example in examples:
        path = os.path.join( coco_train_folder, 'train2017', example[ 'image' ] )
        image = Image.open( path )
        image.load()
        messages.append( _train_conversation_mapper( example, image ) )

    return messages_as_batch( messages, pad_to, processor, assistant_prefix, assistant_suffix )

def _validation_collate_fn(
    examples: list,
    padding_size: int,
    processor: ProcessorMixin,
    assistant_prefix: list[int],
    assistant_suffix: list[int],
):
    pad_to = padding_size + 1

    messages = []

    for example in examples:
        image = example[ 'image' ]
        image.load()
        messages.append( [
            make_multimodal_user_turn_from_images( ( '<image>', '\n' + example[ 'question' ] ), [ image ] ),
            make_multimodal_assistant_turn( example[ 'gpt_answer' ] )
        ] )

        messages.append( [
            make_multimodal_user_turn_from_images( ( example[ 'question' ] + '\n', '<image>' ), [ image ] ),
            make_multimodal_assistant_turn( example[ 'gpt_answer' ] )
        ] )

    return messages_as_batch( messages, pad_to, processor, assistant_prefix, assistant_suffix )

def _evaluation_collate_fn(
    examples: list[dict],
    processor: ProcessorMixin,
):
    message_list = []
    caption_list = []
    for example in examples:
        image = example[ 'image' ]
        image.load()
        
        message_list.append( [
            make_multimodal_user_turn_from_images( ( '<image>', '\n' + example[ 'question' ] ), [ image ] ),
        ] )

        message_list.append( [
            make_multimodal_user_turn_from_images( ( example[ 'question' ] + '\n', '<image>' ), [ image ] ),
        ] )

        caption_list.append( [ example[ 'gpt_answer' ] ] )
        caption_list.append( [ example[ 'gpt_answer' ] ] )
    
    batch: BatchFeature = processor.apply_chat_template(
        message_list,
        tokenize=True,
        return_tensors='pt',
        padding=True,
        padding_side='left',
        truncation=False,
        return_dict=True,
        add_generation_prompt=True,
    ) # type: ignore

    return batch, caption_list

class LlavaInstructDataset( BaseDataset ):
    """ LLaVA Instruct 150k visual QA dataset. """

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
        assert isinstance( train_folder, str )
        self.train_folder = train_folder

        # Download the vqa dataset
        train_dataset = load_dataset( 'json', data_files={ 'train': LLAVA_INSTRUCT_URL }, split='train', cache_dir=cache_dir )
        assert isinstance( train_dataset, Dataset )
        self.train_split = train_dataset

        # Download the benchmark dataset
        valid_dataset = load_dataset( 'lmms-lab/llava-bench-in-the-wild', cache_dir=cache_dir )
        assert isinstance( valid_dataset, DatasetDict )
        self.valid_split = valid_dataset[ 'train' ] # Yes, the validation split is marked as train

    def get_name( self ) -> str:
        return 'llava150k'

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
        )

    def validation_collate_fn( self, examples: list ) -> BatchFeature:
        return _validation_collate_fn(
            examples=examples,
            padding_size=self.sequence_length,
            processor=self.processor,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
        )

    def evaluation_collate_fn( self, examples: list[dict] ) -> tuple[BatchFeature, list[list[str]]]:
        return _evaluation_collate_fn(
            examples=examples,
            processor=self.processor,
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
        longest_example_len = 0
        for example in tqdm.tqdm( self.get_train_split(), desc='Computing train sequence length' ):
            assert isinstance( example, dict )
            messages = _train_conversation_mapper( example, '' )
            example_str = self.processor.apply_chat_template( messages )
            example_len = len( self.tokenizer.encode( example_str ) )
            if example_len > longest_example_len:
                longest_example_len = example_len

        for example in tqdm.tqdm( self.get_validation_split(), desc='Computing validation sequence length' ):
            assert isinstance( example, dict )

            messages_list = []

            messages_list.append( [
                make_multimodal_user_turn( ( '<image>', '\n' + example[ 'question' ] ), [ '' ] ),
                make_multimodal_assistant_turn( example[ 'gpt_answer' ] )
            ] )

            messages_list.append( [
                make_multimodal_user_turn( ( example[ 'question' ] + '\n', '<image>' ), [ '' ] ),
                make_multimodal_assistant_turn( example[ 'gpt_answer' ] )
            ] )

            for messages in messages_list:
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