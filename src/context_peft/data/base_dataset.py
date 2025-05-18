import gc
from collections.abc import Iterator
from abc import ABC, abstractmethod

import torch.utils.data.dataloader as dataloader

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from datasets import Dataset

from .dataset_utils import batched_shuffle_iter

class PrefetchDataset( dataloader.IterableDataset ):
    def __init__( self, dataset: 'BaseDataset', seed_start=0, seed_step=1 ):
        super().__init__()
        self.dataset = dataset
        self.seed_start = seed_start
        self.seed_step = seed_step

    def __iter__( self ):
        gc.collect()
        info = dataloader.get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        for i in self.dataset.train_iterator( num_workers=num_workers, worker_id=worker_id, seed_start=self.seed_start, seed_step=self.seed_step ):
            yield i

class PrefetchValidationDataset( dataloader.IterableDataset ):
    def __init__( self, dataset: 'BaseDataset' ):
        super().__init__()
        self.dataset = dataset

    def __iter__( self ):
        gc.collect()
        for i in self.dataset.validation_iterator():
            yield i

class PrefetchEvaluationDataset( dataloader.IterableDataset ):
    def __init__( self, dataset: 'BaseDataset' ):
        super().__init__()
        self.dataset = dataset

    def __iter__( self ):
        gc.collect()
        for i in self.dataset.evaluation_iterator():
            yield i

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

    @abstractmethod
    def evaluation_collate_fn( self, examples: list[dict] ) -> tuple[BatchFeature, list[list[str]]]:
        """ Constructs a BatchFeature and list of gold targets.

        Args:
            example (list[dict]): A list of evaluation example.

        Returns:
            out (tuple[BatchFeature, list[str]]): A feature dict containing all inputs needed for the forward pass
                and a list of strings corresponding to gold targets.
        """
        raise NotImplementedError()
    
    def train_iterator( self, seed_start=0, seed_step=1, num_workers=1, worker_id=0 ) -> Iterator[BatchFeature]:
        """ Creates an iterator over the dataset which performs shuffling, and yields BatchFeatures

        Args:
            seed_start (int, optional): Starting shuffle seed. Defaults to 0.
            seed_step (int, optional): Step size to increment seed by each epoch. Defaults to 1.
            num_workers (int, optional): The number of workers to split loading across. Defaults to 1.
            worker_id (int, optional): The index of the current worker. Defaults to 0.

        Yields:
            BatchFeature: A feature dict containing all inputs needed for the forward pass and a `labels` item
        """
        for i, row in enumerate( batched_shuffle_iter(
            ds=self.get_train_split(),
            batch_size=self.batch_size,
            seed_start=seed_start,
            seed_step=seed_step
        ) ):
            if i % num_workers == worker_id:
                yield self.train_collate_fn( row )
    
    @abstractmethod
    def validation_iterator( self )  -> Iterator[BatchFeature]:
        """ Creates an iterator over the validation split, yielding BatchFeatures

        Yields:
            BatchFeature: A feature dict containing all inputs needed for the forward pass and a `labels` item
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluation_iterator( self ) -> Iterator[tuple[BatchFeature, list[list[str]]]]:
        """ Creates an iterator over the evaluation split, yielding BatchFeatures and a list of gold target strings.

        Yields:
            out (tuple[BatchFeature, list[str]]]): A feature dict containing all inputs needed for the forward pass
                and a list of strings corresponding to gold targets.
        """
        raise NotImplementedError()

    def train_dataloader( self, num_workers: int, seed_start=0, seed_step=1, **kwargs ) -> dataloader.DataLoader:
        """ Creates a torch DataLoader with parallel support.

        Args:
            num_workers (int): The number of workers to split loading across.
            seed_start (int, optional): Starting shuffle seed. Defaults to 0.
            seed_step (int, optional): Step size to increment seed by each epoch. Defaults to 1.
            kwargs: Any additional kwargs that `DataLoader` accepts apart from batch_size.

        Returns:
            dataloader.DataLoader: Initialised DataLoader
        """
        prefetch_dataset = PrefetchDataset( self, seed_start=seed_start, seed_step=seed_step )

        return dataloader.DataLoader(
            prefetch_dataset,
            batch_size=None,
            num_workers=num_workers,
            **kwargs,
        )

    def validation_dataloader( self, worker: bool, **kwargs ) -> dataloader.DataLoader:
        """ Creates a torch DataLoader with parallel support.

        Args:
            worker (bool): When True the dataloader uses an axuilliary worker process for loading.
            kwargs: Any additional kwargs that `DataLoader` accepts apart from batch_size.

        Returns:
            dataloader.DataLoader: Initialised DataLoader
        """
        prefetch_dataset = PrefetchValidationDataset( self )

        return dataloader.DataLoader(
            prefetch_dataset,
            batch_size=None,
            num_workers=1 if worker else 0,
            **kwargs,
        )

    def evaluation_dataloader( self, worker: bool, **kwargs ) -> dataloader.DataLoader:
        """ Creates a torch DataLoader with parallel support.

        Args:
            worker (bool): When True the dataloader uses an axuilliary worker process for loading.
            kwargs: Any additional kwargs that `DataLoader` accepts apart from batch_size.

        Returns:
            dataloader.DataLoader: Initialised DataLoader
        """
        prefetch_dataset = PrefetchEvaluationDataset( self )

        return dataloader.DataLoader(
            prefetch_dataset,
            batch_size=None,
            num_workers=1 if worker else 0,
            **kwargs,
        )

    @abstractmethod
    def set_optimal_sequence_length( self, pad_to_multiple=32, image_seq_len: int | None = None ) -> tuple[int, int]:
        """ Calculates and sets a more optimal maximum sequence length.

        Args:
            pad_to_multiple (int, optional): Integer multiple to pad to. Defaults to 32.
            image_seq_len (int | None, optional): The (maximum) number of tokens per image. When None we
                will check for `image_seq_len` or `image_seq_length` in the Processor. Defaults to None.

        Returns:
            unpadded_sequence_length (int): The unpadded sequence length requried by the dataset.
            padded_sequence_length (int): The unpadded sequence length padded to the next multiple.
        """
        raise NotImplementedError()