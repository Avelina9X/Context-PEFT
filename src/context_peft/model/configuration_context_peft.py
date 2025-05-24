from typing import Any, Literal
from types import NoneType
from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig, CONFIG_MAPPING


def check_types( value, types: tuple ) -> bool:
    """ Checks that a value adheres to dict schema.

    The `types` parameter accepts a tuple which may contain the following elemnts:
    - Any generic type (e.g. `str`, `int`, `bool`, etc)
    - A list parameterised by a generic type (e.g. `list[str]`)
    - A Literal type for specifc values (e.g. `Literal['auto']`)
    - The `NoneType` to allow `None` as an acceptable value

    Args:
        value (Any): The value to check
        types (tuple): A tuple of allowed types

    Returns:
        bool: Returns True iff the value matches any type in the tuple
    """
    for t in types:
        origin_type = getattr( t, '__origin__', t )
        if origin_type is Literal:
            if any( value == a for a in t.__args__ ):
                return True
        elif isinstance( value, origin_type ):
            if isinstance( value, list ) and issubclass( origin_type, list ):
                if all( isinstance( v, t.__args__[0] ) for v in value ):
                    return True
            else:
                return True
    return False

def validate_peft_config( config: dict, peft_type: str, adaptor_name: str ):
    """ Validates the config for an individual adaptor.

    Args:
        config (dict): The config dict of a single adaptor
        peft_type (str): The adaptor type, used to select the correct schema
        adaptor_name (str): The adaptor name, used to create informative errors
    """

    # The config requires a 'type' key to ensure the adaptor config matches the model's PEFT type
    if 'type' not in config:
        raise ValueError( f'Key "type" not present in PEFT config for adaptor "{adaptor_name}"!' )
    if config[ 'type' ] != peft_type:
        raise ValueError( f'Expected type:{peft_type} but found "{config["type"]}" in PEFT config for adaptor "{adaptor_name}"!' )

    # Type schema for the LoRA, IA3 and BitFit configs
    required_args_map = {
        'lora': {
            'context': ( str, list[str] ),
            'r': ( int, ),
            'target_modules': ( str, list[str] ),
            'exclude_modules': ( str, list[str], NoneType ),
            'lora_alpha': ( int, ),
            'use_rslora': ( bool, ),
            'use_bias': ( bool, Literal['auto'] )
        },
        'ia3': {
            'context': ( str, list[str] ),
            'target_modules': ( str, list[str] ),
            'exclude_modules': ( str, list[str], NoneType ),
            'feedforward_modules': ( str, list[str], NoneType ),
        },
        'bitfit':{
            'context': ( str, list[str] ),
            'target_modules': ( str, list[str] ),
            'exclude_modules': ( str, list[str], NoneType ),
            'force_bias': ( bool, ),
        }
    }

    required_args = required_args_map[ peft_type ]

    for key, types in required_args.items():
        if key not in config:
            raise ValueError( f'Key "{key}" not present in PEFT config for adaptor "{adaptor_name}"!' )
        if not check_types( config[ key ], types ):
            raise ValueError( f'Invalid value for "{key}" in PEFT config for adaptor "{adaptor_name}"!' )

    context = config[ 'context' ]
    if isinstance( context, str ):
        context = [ context ]

    for c in context:
        if c not in [ 'text', 'image' ]:
            raise ValueError( f'Invalid context type "{c}" found in PEFT config for adaptor "{adaptor_name}"!' )

def update_peft_config( config: dict, default: dict | None ):
    """ Updates adaptor config in place using defaults

    Args:
        config (dict): The config dict of a single adaptor 
        default (dict | None): Default config dict, if specified
    """
    if default is not None:
        for key, value in default.items():
            if key not in config:
                config[key] = value

class ContextPeftConfig( PretrainedConfig ):
    model_type = 'context_peft'

    sub_configs = {
        'text_config': AutoConfig, # type: ignore
        'vision_config': AutoConfig, # type: ignore
    }

    def __init__(
        self,

        vision_config: PretrainedConfig | dict | None = None,
        vision_dim: int = 768,
        vision_trainable: bool = False,

        text_config: PretrainedConfig | dict | None = None,
        text_dim: int = 1024,
        text_trainable: bool = False,

        image_pad_token_id: int = 151647,

        connector_activation: str = 'gelu',
        connector_bias: bool = False,
        connector_trainable: bool = True,
        connector_dropout: float = 0.0,

        peft_type: str | None = None,
        default_peft_config: dict | None = None,
        adaptors: dict | None = None,
        active_adaptors: list | None = None,
        adaptor_dropout: float = 0.0,

        additional_adaptors: dict | None = None,
        additional_active_adaptors: list | None = None,

        use_cache=True,
        **kwargs,
    ):
        # Reset all adaptors and adaptor map for auto-infer
        kwargs.pop( 'all_adaptors', None )
        kwargs.pop( 'adaptor_map', None )

        self.use_cache = use_cache

        # Instantiate vision config from dict
        if vision_config is None:
            vision_config = PretrainedConfig()
        elif isinstance( vision_config, dict ):
            vision_config = CONFIG_MAPPING[ vision_config[ 'model_type' ] ]( **vision_config )
            assert isinstance( vision_config, PretrainedConfig )

        # Instantiate text config from dict
        if text_config is None:
            text_config = PretrainedConfig()
        elif isinstance( text_config, dict ):
            text_config = CONFIG_MAPPING[ text_config[ 'model_type' ] ]( **text_config )
            assert isinstance( text_config, PretrainedConfig )

        self.vision_config = vision_config
        self.vision_dim = vision_dim
        self.vision_trainable = vision_trainable

        self.text_config = text_config
        self.text_dim = text_dim
        self.text_trainable = text_trainable

        self.image_pad_token_id = image_pad_token_id

        self.connector_activation = connector_activation
        self.connector_bias = connector_bias
        self.connector_trainable = connector_trainable
        self.connector_dropout = connector_dropout

        self.peft_type = peft_type
        self.default_peft_config = default_peft_config
        self.adaptors = deepcopy( adaptors or {} )
        self.adaptor_map = {}
        self.adaptor_dropout = adaptor_dropout

        # Add additional adaptors, useful for setting additional kwargs when initialising from an existing config
        if additional_adaptors is not None:
            if overwritten_adaptors := set( self.adaptors ) & set( additional_adaptors ):
                raise ValueError( f'Attempting to add adaptors with existing names: {overwritten_adaptors}' )
            self.adaptors.update( additional_adaptors )

            if additional_active_adaptors is None:
                additional_active_adaptors = list( additional_adaptors.keys() )

        # Full expand out partial adaptor configs and validate
        for adaptor_name, adaptor_config in self.adaptors.items():
            if peft_type is None:
                raise ValueError( 'Cannot use any adaptors when peft_type is None!' )
            update_peft_config( adaptor_config, default_peft_config )
            validate_peft_config( adaptor_config, peft_type, adaptor_name )

        # Get all and active adaptors
        self.all_adaptors = list( dict.fromkeys( self.adaptors ) )
        self.active_adaptors = active_adaptors or self.all_adaptors

        # Add additional active adaptors, useful for setting additional kwargs when initialising from an existing config
        if additional_active_adaptors is not None:
            self.active_adaptors = list( dict.fromkeys( self.active_adaptors + additional_active_adaptors ) )

        # Check if unknown adaptors have been added
        if unknown_adaptors := set( self.active_adaptors ) - set( self.all_adaptors ):
            raise ValueError( f'Unknown adaptors {unknown_adaptors} set in active_adaptors! Must be one of {self.all_adaptors}.' )

        # Update adaptor map based on adaptor context type
        for adaptor_name, adaptor_config in self.adaptors.items():
            if adaptor_name in self.active_adaptors:
                contexts = adaptor_config[ 'context' ]
                if isinstance( contexts, str ):
                    contexts = [ contexts ]
                self.adaptor_map[ adaptor_name ] = contexts

        # When peft_type is none there should be no adaptors!
        if peft_type is None and ( self.adaptors or self.all_adaptors or self.active_adaptors ):
            raise ValueError( 'Cannot use any adaptors when peft_type is None!' )
        if peft_type is None and default_peft_config:
            raise ValueError( 'Cannot set a default peft config when peft_type is None!' )

        super().__init__( **kwargs )

    @classmethod
    def from_dict( cls, config_dict: dict[str, Any], **kwargs ) -> PretrainedConfig:
        graft_kwargs = [
            'peft_type',
            'default_peft_config',
            'adaptors',
            'active_adaptors',
            'additional_adaptors',
            'additional_active_adaptors',
        ]
        
        for key in graft_kwargs:
            if key in kwargs:
                config_dict[key] = kwargs.pop( key )
        
        return super().from_dict( config_dict, **kwargs )