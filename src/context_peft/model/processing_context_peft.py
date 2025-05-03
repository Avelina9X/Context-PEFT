from transformers.tokenization_utils_base import AddedToken, TextInput, PreTokenizedInput, PreTrainedTokenizerBase
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ImagesKwargs, Unpack, ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_nested_list_of_images

CHAT_TEMPLATE_MAP = {
    'chat_ml': (
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}"
        "{{ '<|im_start|>' + message['role'] + '\n' }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}"
        "{{ '<|cpeft_image_placeholder|>' }}"
        "{% elif content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    ),
    'chat_ml_bos': (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}"
        "{{ '<|im_start|>' + message['role'] + '\n' }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}"
        "{{ '<|cpeft_image_placeholder|>' }}"
        "{% elif content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    ),
}


class ContextPeftProcessorKwargs( ProcessingKwargs, total=False ):
    images_kwargs: ImagesKwargs
    _defaults = {
        "text_kwargs": {}, # type: ignore
        "images_kwargs": {}, # type: ignore
    }

class ContextPeftProcessor( ProcessorMixin ):
    attributes = [ 'image_processor', 'tokenizer' ]
    valid_kwargs = [ 'image_seq_len', 'chat_template' ]
    image_processor_class = 'AutoImageProcessor'
    tokenizer_class = 'AutoTokenizer'

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: PreTrainedTokenizerBase,
        image_seq_len: int,
        chat_template: str,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_seq_len = image_seq_len

        self.image_placeholder_token = AddedToken( '<|cpeft_image_placeholder|>', normalized=False, special=True )
        self.image_pad_token = AddedToken( '<|cpeft_image_pad|>', normalized=False, special=True )

        tokenizer.add_special_tokens( {
            'additional_special_tokens': [
                self.image_placeholder_token, # type: ignore
                self.image_pad_token, # type: ignore
            ]
        } )

        if chat_template in CHAT_TEMPLATE_MAP.keys():
            chat_template = CHAT_TEMPLATE_MAP[ chat_template ]

        super().__init__( image_processor, tokenizer, chat_template=chat_template, **kwargs )

    def get_image_pad_token_id( self ):
        """ Returns the token id of the image padding token """
        return self.tokenizer.convert_tokens_to_ids( self.image_pad_token.content ) # type: ignore

    def __call__(
        self,
        images: ImageInput | list[ImageInput] | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos = None,
        audio = None,
        **kwargs: Unpack[ContextPeftProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            ContextPeftProcessorKwargs, # type: ignore
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is None:
            raise ValueError( 'Text must be specified!' )

        n_images_in_text = []
        n_images_in_images = []
        inputs = BatchFeature()

        if isinstance( text, str ):
            text = [ text ]
        elif not isinstance( text, list ) and not isinstance( text[0], str ):
            raise ValueError( 'Invalid input text. Please provide a string, or a list of strings' )
        n_images_in_text = [ sample.count( self.image_placeholder_token.content ) for sample in text ]

        if images is not None:
            batched_images = make_nested_list_of_images( images )
            image_inputs = self.image_processor( batched_images, **output_kwargs["images_kwargs"] )
            inputs.update(image_inputs)

            n_images_in_images = [ len( sample ) for sample in images ] # type: ignore


        if sum( n_images_in_images ) != sum( n_images_in_text ):
            raise ValueError(
                f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
            )

        prompt_strings = [ sample.replace( self.image_placeholder_token.content, self.image_pad_token.content * self.image_seq_len ) for sample in text ] # type: ignore
        text_inputs = self.tokenizer( text=prompt_strings,  **output_kwargs["text_kwargs"] )
        inputs.update(text_inputs)

        return inputs
