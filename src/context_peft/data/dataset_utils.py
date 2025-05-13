from collections.abc import Iterator
from collections import Counter
from itertools import count
import re
import string

from PIL import Image
import torch
from datasets import Dataset

def compute_assistant_mask(
    input_ids: torch.Tensor,
    assistant_prefix: list[int],
    assistant_suffix: list[int],
    ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    """ Computes the left-shifted labels for calculating loss on assistant tokens only.

    IMPORTANT! input_ids should be padded to ***sequence_length + 1*** as the
    final token will be truncated due to computing the left shifted labels.

    The masked labels will have a value of `ignore_index` outside of the assistant token span.
    The unmasked label spans start **after** the final assistant_prefix token and **includes**
    the final assistant_suffix token (i.e. span excludes the prefix and includes the suffix).

    This method will correctly mask all assistant spans within a sequence, and also works for
    sequences where the final assistant suffix has been truncated.

    Args:
        input_ids (torch.Tensor): input_ids of size [batch, sequence_length+1]
        assistant_prefix (list[int]): list of tokens to match for assistant prefix
        assistant_suffix (list[int]): list of tokens to match for assistant suffix
        ignore_index (int, optional): the id to use for masked tokens. Defaults to -100.

    Returns:
        input_ids (torch.Tensor): truncated input_ids of size [batch, sequence_length]
        labels (torch.Tensor): left shifted, masked labels of size [batch, sequence_length]
    """

    start_mask = True
    for i, token in enumerate( assistant_prefix ):
        id_mask = input_ids.roll( -i + len( assistant_prefix ) - 1 ) == token
        start_mask *= id_mask

    end_mask = True
    for i, token in enumerate( assistant_suffix ):
        id_mask = input_ids.roll( -i + len( assistant_suffix ) - 1 ) == token
        end_mask *= id_mask

    tok_range = torch.arange( input_ids.shape[-1] )

    mask = ( tok_range * start_mask ).cummax( -1 )[0] > ( tok_range * end_mask ).cummax( -1 )[0]

    new_input_ids = input_ids[ :, : -1 ]
    target_ids = input_ids[ :, 1 : ]
    mask = mask[ :, : -1 ]

    target_ids = torch.where( mask, target_ids, ignore_index )

    return new_input_ids, target_ids

def shuffle_iter( ds: Dataset, seed_start=0, seed_step=1 ) -> Iterator:
    """ Creates an infinite iterator over a dataset, suffling each repeat

    Args:
        ds (Dataset): Hugging face dataset
        seed_start (int, optional): Starting shuffle seed. Defaults to 0.
        seed_step (int, optional): Step size to increment seed by. Defaults to 1.

    Yields:
        dict: A row from the dataset
    """
    for i in count( seed_start, seed_step ):
        for row in ds.shuffle( seed=i ):
            yield row

def batched_shuffle_iter( ds: Dataset, batch_size: int, seed_start=0, seed_step=1 ) -> Iterator[list]:
    """ Creates an infinite iterator over a dataset, suffling each repeat, yielding a batch of rows

    Args:
        ds (Dataset): Hugging face dataset
        batch_size (int): Batch size
        seed_start (int, optional): Starting shuffle seed. Defaults to 0.
        seed_step (int, optional): Step size to increment seed by. Defaults to 1.

    Yields:
        list[dict]: A batch of dataset rows
    """
    batch = []
    for row in shuffle_iter( ds, seed_start=seed_start, seed_step=seed_step ):
        batch.append( row )

        if len( batch ) == batch_size:
            yield batch
            batch = []

def make_multimodal_user_turn( sub_strings: tuple[str, ...], paths: list[str] ) -> dict:
    """ Creates a single chat message turn from string parts and an image path.

    All strings within `sub_strings` will be treated as type text, except for a string
    exactly matching `"<image>"` which will become type image with the given path. 

    Args:
        sub_strings (tuple[str, ...]): A tuple of strings defining the message structure. 
        paths (list[str]): List of image paths.

    Returns:
        dict: A single user message turn to be added to a conversation list.
    """
    paths = paths.copy()
    contents = []
    for string in sub_strings:
        if string == '<image>':
            contents.append( { 'type': 'image', 'path': paths.pop( 0 ) } )
        else:
            contents.append( { 'type': 'text', 'text': string } )
    
    if len( paths ) != 0:
        raise ValueError( 'Missmatch between number of images and number of paths!' )
    
    return {
        'role': 'user',
        'content': contents,
    }

def make_multimodal_user_turn_from_images( sub_strings: tuple[str, ...], paths: list[Image.Image] ) -> dict:
    """ Creates a single chat message turn from string parts and a PIL image.

    All strings within `sub_strings` will be treated as type text, except for a string
    exactly matching `"<image>"` which will become type image with the given image. 

    Args:
        sub_strings (tuple[str, ...]): A tuple of strings defining the message structure. 
        paths (list[Image]): List of images.

    Returns:
        dict: A single user message turn to be added to a conversation list.
    """
    paths = paths.copy()
    contents = []
    for string in sub_strings:
        if string == '<image>':
            contents.append( { 'type': 'image', 'image': paths.pop( 0 ) } )
        else:
            contents.append( { 'type': 'text', 'text': string } )
    
    if len( paths ) != 0:
        raise ValueError( 'Missmatch between number of images and number of paths!' )
    
    return {
        'role': 'user',
        'content': contents,
    }

def make_multimodal_assistant_turn( caption: str ) -> dict:
    """ Creates a single chat message turn from a caption

    Args:
        caption (str): Assistant message contents

    Returns:
        dict: A single assistant message turn to be added to a conversation list.
    """
    return {
        'role': 'assistant',
        'content': [ { 'type': 'text', 'text': caption } ],
    }

ARTICLES_REGEX = re.compile( r"\b(a|an|the)\b", re.UNICODE )

def normalize_answer( s: str ):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles( text: str ):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix( text: str ):
        return " ".join(text.split())

    def remove_punc( text: str ):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower( text: str ):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1( pred: str, gold: list[str] ):
    best_f1 = 0

    pred_toks = normalize_answer( pred ).split()

    for gold_str in gold:
        gold_toks = normalize_answer( gold_str ).split()
        common = Counter( gold_toks ) & Counter( pred_toks )
        num_same = sum( common.values() )
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len( pred_toks )
        recall = 1.0 * num_same / len( gold_toks )
        f1 = ( 2 * precision * recall ) / ( precision + recall )

        best_f1 = max( best_f1, f1 )
    
    return best_f1

