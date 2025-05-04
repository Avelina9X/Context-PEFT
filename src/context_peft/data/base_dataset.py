from collections.abc import Iterator
from abc import ABC, abstractmethod

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from datasets import Dataset

from .dataset_utils import batched_shuffle_iter


class BaseDataset( ABC ):
    """ Base dataset class for multimodal image-text datasets.

    All subclasses must implement `get_train_split()` and `collate_fn()`.
    """
    
    def __init__(
        self,
        processor: ProcessorMixin,
        assistant_prefix: list[int] | str,
        assistant_suffix: list[int] | str,
        batch_size: int,
        sequence_length: int,
    ):
        """ Instantiates a generic multimodal image-text dataset

        Args:
            processor (ProcessorMixin): Input processor with a tokenizer, image processor and chat template support.
            assistant_prefix (list[int] | str): Prefix of assistant messages. May be a string or list of token ids.
            assistant_suffix (list[int] | str): Suffix of assistant messages. May be a string or list of token ids.
            batch_size (int): Training batch size.
            sequence_length (int): Sequence length, will pad all sequences up to this size.
        """

        self.processor = processor
        self.tokenizer: PreTrainedTokenizerBase = getattr( processor, 'tokenizer' )
        
        if isinstance( assistant_prefix, str ):
            assistant_prefix = self.tokenizer.encode( assistant_prefix, add_special_tokens=False )
        if isinstance( assistant_suffix, str ):
            assistant_suffix = self.tokenizer.encode( assistant_suffix, add_special_tokens=False )
        
        self.assistant_prefix = assistant_prefix
        self.assistant_suffix = assistant_suffix

        self.batch_size = batch_size
        self.sequence_length = sequence_length
    
    @abstractmethod
    def get_train_split( self ) -> Dataset:
        """ Returns the pre-mapped training split """
        raise NotImplementedError()
    
    @abstractmethod
    def get_validation_split( self ) -> Dataset:
        """ Returns the pre-mapped validation split """
        raise NotImplementedError()
    
    @abstractmethod
    def train_collate_fn( self, examples: list ) -> BatchFeature:
        """ Constructs a BatchFeature from a list of dataset rows.

        Args:
            examples (list): List which forms the batch

        Returns:
            BatchFeature: A feature dict containing all inputs needed for the forward pass and a `labels` item
        """
        raise NotImplementedError()
    
    @abstractmethod
    def validation_collate_fn( self, examples: list ) -> BatchFeature:
        """ Constructs a BatchFeature from a list of dataset rows.

        Args:
            examples (list): List which forms the batch

        Returns:
            BatchFeature: A feature dict containing all inputs needed for the forward pass and a `labels` item
        """
        raise NotImplementedError()
    
    def train_iterator( self, seed_start=0, seed_step=1 ) -> Iterator[BatchFeature]:
        """ Creates an iterator over the dataset which performs shuffling, and yields BatchFeatures

        Args:
            seed_start (int, optional): Starting shuffle seed. Defaults to 0.
            seed_step (int, optional): Step size to increment seed by. Defaults to 1.

        Yields:
            BatchFeature: A feature dict containing all inputs needed for the forward pass and a `labels` item
        """
        for row in batched_shuffle_iter(
            ds=self.get_train_split(),
            batch_size=self.batch_size,
            seed_start=seed_start,
            seed_step=seed_step
        ):
            yield self.train_collate_fn( row )
    
    @abstractmethod
    def validation_iterator( self )  -> Iterator[BatchFeature]:
        """ Creates an iterator over the validation split, yielding BatchFeatures

        Yields:
            BatchFeature: A feature dict containing all inputs needed for the forward pass and a `labels` item
        """

        raise NotImplementedError()
