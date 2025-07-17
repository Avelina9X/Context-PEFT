from typing import Literal
from types import NoneType

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

    common_args_map = {
        'modalities': ( list[str], NoneType ),
        'messages': ( list[str], NoneType ),
        'target_modules': ( str, list[str] ),
        'exclude_modules': ( str, list[str], NoneType ),
    }

    # Type schema for the LoRA, IA3 and BitFit configs
    required_args_map = {
        'lora': {
            'r': ( int, ),
            'lora_alpha': ( int, ),
            'use_rslora': ( bool, ),
            'use_bias': ( bool, Literal['auto'] ),
            'initialization': ( str, ),
            'scale_multiplier': ( float, ),
            **common_args_map,
        },
        'ia3': {
            'feedforward_modules': ( str, list[str], NoneType ),
            **common_args_map,
        },
        'bitfit':{
            'force_bias': ( bool, ),
            **common_args_map,
        }
    }

    required_args = required_args_map[ peft_type ]

    for key, types in required_args.items():
        if key not in config:
            raise ValueError( f'Key "{key}" not present in PEFT config for adaptor "{adaptor_name}"!' )
        if not check_types( config[ key ], types ):
            raise ValueError( f'Invalid value for "{key}" in PEFT config for adaptor "{adaptor_name}"!' )

    modalities = config[ 'modalities' ]
    messages = config[ 'messages' ]

    if modalities is not None:
        for c in modalities:
            if c not in [ 'text', 'image' ]:
                raise ValueError( f'Invalid modality type "{c}" found in PEFT config for adaptor "{adaptor_name}"!' )

    if messages is not None:
        for c in messages:
            if c not in [ 'system', 'user', 'assistant' ]:
                raise ValueError( f'Invalid message type "{c}" found in PEFT config for adaptor "{adaptor_name}"!' )

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