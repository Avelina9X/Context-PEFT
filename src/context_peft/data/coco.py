from collections.abc import Iterator
import os
import math
import random
import aiohttp

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
    make_multimodal_user_turn
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

        messages.append( [
            make_multimodal_user_turn( prompt, [ path ] ),
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
    coco_validation_folder: str,
    padding_size: int,
    processor: ProcessorMixin,
    assistant_prefix: list[int],
    assistant_suffix: list[int],
    caption_instruction: tuple[str, ...],
):
    pad_to = padding_size + 1

    messages = []

    for example in examples:
        path = os.path.join( coco_validation_folder, example[ 'file_name' ] )
        for caption in example[ 'captions' ]:
            messages.append( [
                make_multimodal_user_turn( caption_instruction, [ path ] ),
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

        # Apply map to split captions
        self.train_split = _split_captions( dataset[ 'train' ] )

        # Set validation split as is
        self.valid_split = dataset[ 'validation' ]
    
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
            caption_instruction_map=CAPTION_INSTRUCTIONS_MAP
        )
    
    def validation_collate_fn( self, examples: list ) -> BatchFeature:
        return _validation_collate_fn(
            examples=examples,
            coco_validation_folder=self.valid_folder,
            padding_size=self.sequence_length,
            processor=self.processor,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
            caption_instruction=CAPTION_INSTRUCTIONS_MAP[0]
        )
    
    def validation_iterator(self) -> Iterator[BatchFeature]:
        for row in self.get_validation_split():
            yield self.validation_collate_fn( [ row ] )


def calculate_padding( vision_model_name: str, pad_to_multiple=32 ) -> int:
    """ Calculates the recommended COCO sequence for Context-PEFT defaults.

    IMPORTANT: It is ***not*** recommended to use this if you aren't running
    the standard experiments. Changing any conifg or processor settings will
    not give you correct padding estimates! Use with caution!

    Args:
        vision_model_name (str): Name of the openai CLIP variant.
        pad_to_multiple (int, optional): Integer multiple to pad to. Defaults to 32.

    Returns:
        int: Recommended max sequence length.
    """
    padding_map = {
        'openai/clip-vit-base-patch32': 152,
        'openai/clip-vit-base-patch16': 299,
        'openai/clip-vit-large-patch14': 359,
        'openai/clip-vit-large-patch14-336': 659,
    }

    return math.ceil( padding_map[ vision_model_name ] / pad_to_multiple ) * pad_to_multiple