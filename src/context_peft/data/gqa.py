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
    EvalAIAnswerProcessor,
    compute_assistant_mask,
    compute_f1,
    make_multimodal_assistant_turn,
    make_multimodal_user_turn,
    make_multimodal_user_turn_from_images,
)

GQA_IMAGE_URL = 'https://huggingface.co/datasets/lmms-lab/GQA/resolve/main/testdev_balanced_images/testdev-00000-of-00001.parquet'
GQA_QUESTION_URL = 'https://huggingface.co/datasets/lmms-lab/GQA/resolve/main/testdev_balanced_instructions/testdev-00000-of-00001.parquet'

def _evaluation_collate_fn(
    examples: list[dict],
    processor: ProcessorMixin,
    test_image_cache: dict,
):
    message_list = []
    caption_list = []
    for example in examples:
        image = test_image_cache[ example[ 'imageId' ] ]

        question = example[ 'question' ]
        prompt = f'\n{question}\nAnswer the question using a single word or phrase.'
        
        message_list.append( [
            make_multimodal_user_turn_from_images( ( '<image>', prompt ), [ image ] ),
        ] )

        caption_list.append( [ example[ 'answer' ], example[ 'fullAnswer' ] ] )
    
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

class GQADataset( BaseDataset ):
    """ GQA visual reasoning dataset. """

    def __init__(
        self,
        processor: ProcessorMixin,
        assistant_prefix: list[int] | str,
        assistant_suffix: list[int] | str,
        batch_size: int,
        sequence_length: int,
        cache_dir: str | None = None
    ):
        """ Instantiates a GQA dataset.

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

        gqa_test_images = load_dataset( 'parquet', data_files={ 'test': GQA_IMAGE_URL }, split='test', cache_dir=cache_dir )
        gqa_test_questions = load_dataset( 'parquet', data_files={ 'test': GQA_QUESTION_URL }, split='test', cache_dir=cache_dir )
        gqa_test_questions = gqa_test_questions.select_columns( [ 'imageId', 'question', 'answer', 'fullAnswer' ] )

        assert isinstance( gqa_test_images, Dataset )
        assert isinstance( gqa_test_questions, Dataset )

        self.test_image_cache = {}
        for row in tqdm.tqdm( gqa_test_images, desc='Loading GQA test images' ):
            id = row[ 'id' ]
            image = row[ 'image' ]
            image.load()
            self.test_image_cache[id] = image
        
        self.valid_split = gqa_test_questions

    def get_name( self ) -> str:
        return 'gqa'
    
    def get_train_split( self ) -> Dataset:
        raise NotImplementedError()
    
    def get_validation_split( self ) -> Dataset:
        return self.valid_split

    def train_collate_fn( self, examples: list ) -> BatchFeature:
        raise NotImplementedError()

    def validation_collate_fn( self, examples: list ) -> BatchFeature:
        raise NotImplementedError()

    def evaluation_collate_fn( self, examples: list[dict] ) -> tuple[BatchFeature, list[list[str]]]:
        return _evaluation_collate_fn(
            examples=examples,
            processor=self.processor,
            test_image_cache=self.test_image_cache
        )

    def validation_iterator( self ) -> Iterator[BatchFeature]:
        raise NotImplementedError()

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
        eval_ai_processor: EvalAIAnswerProcessor = kwargs[ 'eval_ai_processor' ]

        assert len( references ) == 2
        
        pred = eval_ai_processor( prediction )
        answer = eval_ai_processor( references[0] )
        full_answer = eval_ai_processor( references[1] )

        exact_match = float( pred == answer )

        f1, precision, recall = compute_f1( pred, [ answer, full_answer ] )

        return { 'exact_match': exact_match, 'f1': f1 }

    def compute_scores( self, predictions: list[str], references: list[list[str]], **kwargs ) -> dict[str, float]:
        return super().compute_scores( predictions, references, eval_ai_processor=EvalAIAnswerProcessor() )

    def set_optimal_sequence_length( self, pad_to_multiple=32, image_seq_len: int | None = None ) -> tuple[int, int]:
        raise NotImplementedError()