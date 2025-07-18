from typing import Any
from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig, CONFIG_MAPPING

from .configuration_utils import validate_peft_config, update_peft_config

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
        adaptor_dropout: float | dict[str, float] | None = None,

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
                self.adaptor_map[ adaptor_name ] = {
                    'modalities': adaptor_config[ 'modalities' ],
                    'messages': adaptor_config[ 'messages' ],
                }

        # When peft_type is none there should be no adaptors!
        if peft_type is None and ( self.adaptors or self.all_adaptors or self.active_adaptors ):
            raise ValueError( 'Cannot use any adaptors when peft_type is None!' )
        if peft_type is None and default_peft_config:
            raise ValueError( 'Cannot set a default peft config when peft_type is None!' )

        if peft_type is not None:
            if adaptor_dropout is None:
                adaptor_dropout = 0.0

            if isinstance( adaptor_dropout, float ):
                adaptor_dropout = { k: adaptor_dropout if k in self.active_adaptors else 0.0 for k in self.all_adaptors }
            elif isinstance( adaptor_dropout, dict ):
                adaptor_dropout = { k: adaptor_dropout[k] if k in adaptor_dropout else 0.0 for k in self.all_adaptors }
            else:
                raise ValueError( 'adaptor_dropout has unknown type!' )
        else:
            if adaptor_dropout is not None:
                raise ValueError( 'adaptor_dropout must be None when peft_type is None!')

        self.adaptor_dropout = adaptor_dropout
            
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
            'adaptor_dropout'
        ]
        
        for key in graft_kwargs:
            if key in kwargs:
                config_dict[key] = kwargs.pop( key )
        
        return super().from_dict( config_dict, **kwargs )