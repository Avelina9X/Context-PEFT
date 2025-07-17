from collections.abc import Iterator
from io import BytesIO
import os
import math
import random
import base64
from collections import defaultdict
import statistics

import json
from typing import DefaultDict
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
    make_multimodal_assistant_turn,
    make_multimodal_user_turn,
    make_multimodal_user_turn_from_images,
    EvalAIAnswerProcessor,
    compute_f1
)

VQA_IMAGE_URL = 'http://images.cocodataset.org/zips/val2014.zip'
VQA_QUESTION_URL = 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip'
VQA_ANNOTATION_URL = 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip'

def _evaluation_collate_fn(
    examples: list[dict],
    processor: ProcessorMixin,
    valid_image_cache: dict,
):
    message_list = []
    caption_list = []
    for example in examples:
        file_path = valid_image_cache[ example[ 'image_id' ] ]
        image = Image.open( file_path )
        image.load()

        question = example[ 'question' ]
        prompt = f'\n{question}\nAnswer the question using a single word or phrase.'
        
        message_list.append( [
            make_multimodal_user_turn_from_images( ( '<image>', prompt ), [ image ] ),
        ] )

        caption_list.append( [ i[ 'answer' ] for i in example[ 'answers' ] ] )
    
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

class VQADataset( BaseDataset ):
    """ VQA questions answering dataset. """

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
        """ Instantiates a VQA dataset.

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
        dl = DownloadManager( 'vqav2_local', download_config=dl_config )

        valid_questions = dl.download_and_extract( VQA_QUESTION_URL )
        valid_annotations = dl.download_and_extract( VQA_ANNOTATION_URL )
        valid_images = dl.download_and_extract( VQA_IMAGE_URL )

        assert isinstance( valid_questions, str )
        assert isinstance( valid_annotations, str )
        assert isinstance( valid_images, str )

        with open( os.path.join( valid_questions, 'v2_OpenEnded_mscoco_val2014_questions.json' ), mode='r' ) as f:
            valid_questions_obj = json.load( f )[ 'questions' ]

        question_ids = {}
        for row in valid_questions_obj:
            question_ids[ row[ 'question_id' ] ] = row[ 'question' ]
        
        with open( os.path.join( valid_annotations, 'v2_mscoco_val2014_annotations.json' ), mode='r' ) as f:
            valid_annotations_obj = json.load( f )[ 'annotations' ]
        vqav2_dataset = Dataset.from_list( valid_annotations_obj ).map( lambda x: { 'question': question_ids[ x[ 'question_id' ] ] } )

        self.valid_image_cache = {}
        for row in tqdm.tqdm( vqav2_dataset, desc='Parsing VQA validation images' ):
            image_id = row[ 'image_id' ]

            if image_id not in self.valid_image_cache:
                file_path = os.path.join( valid_images, 'val2014', f'COCO_val2014_{image_id:0>12}.jpg' )
                # image = Image.open( file_path )
                # image.load()
                self.valid_image_cache[ image_id ] = file_path
        
        self.valid_split = vqav2_dataset

    def get_name( self ) -> str:
        return 'vqav2'

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
            valid_image_cache=self.valid_image_cache
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
        doc = {
            'answers': [ { 'answer': r } for r in references ]
        }

        eval_ai_processor: EvalAIAnswerProcessor = kwargs[ 'eval_ai_processor' ]
        resAns = prediction
        accuracy = 0
        f1 = 0

        if "answers" in doc and doc["answers"] is not None:
            for ansDic in doc["answers"]:
                ansDic["answer"] = ansDic["answer"].replace("\n", " ")
                ansDic["answer"] = ansDic["answer"].replace("\t", " ")
                ansDic["answer"] = ansDic["answer"].strip()
            gtAcc = []
            gtAnswers = [ans["answer"] for ans in doc["answers"]]

            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()

            if len(set(gtAnswers)) > 1:
                for ansDic in doc["answers"]:
                    ansDic["answer"] = eval_ai_processor.process_punctuation(ansDic["answer"])
                    ansDic["answer"] = eval_ai_processor.process_digit_article(ansDic["answer"])
                resAns = eval_ai_processor.process_punctuation(resAns)
                resAns = eval_ai_processor.process_digit_article(resAns)

            for gtAnsDatum in doc["answers"]:
                otherGTAns = [item for item in doc["answers"] if item != gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            accuracy = statistics.mean(gtAcc)

            f1, _, _ = compute_f1( resAns, [ item["answer"] for item in doc["answers"] ] )

        return { 'accuracy': accuracy, 'f1': f1 }

    def compute_scores( self, predictions: list[str], references: list[list[str]], **kwargs ) -> dict[str, float]:
        return super().compute_scores( predictions, references, eval_ai_processor=EvalAIAnswerProcessor() )

    def set_optimal_sequence_length( self, pad_to_multiple=32, image_seq_len: int | None = None ) -> tuple[int, int]:
        raise NotImplementedError()