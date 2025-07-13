from collections.abc import Iterator
from collections import Counter
from itertools import count
import re
import string

from PIL import Image
import torch
from datasets import Dataset

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

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
    best_precision = 0
    best_recall = 0

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
        best_precision = max( best_precision, precision )
        best_recall = max( best_recall, recall )
    
    return best_f1, best_precision, best_recall

def messages_as_batch(
    messages: list,
    pad_to: int,
    processor: ProcessorMixin,
    assistant_prefix: list[int],
    assistant_suffix: list[int]
) -> BatchFeature:
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

import re


class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (re.search(self.COMMA_STRIP, in_text) is not None):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item