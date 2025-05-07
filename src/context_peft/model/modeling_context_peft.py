from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn as nn

from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel, GenerationMixin
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN

from .configuration_context_peft import ContextPeftConfig


class ContextPeftAdaptorBase( ABC ):
    # All names of layers that may contain adapter (trainable) weights
    adaptor_layer_names: tuple[str, ...]
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...]

    # All nn.Module classes the adaptor can wrap
    module_types: tuple[type[nn.Module]]

    def get_base_layer( self ) -> nn.Module:
        """ Recursively get the base_layer. """
        base_layer = self
        while hasattr( base_layer, 'base_layer' ):
            base_layer = base_layer.base_layer # type: ignore
        return base_layer # type: ignore

    @abstractmethod
    def init_adaptor_weights( self ):
        """ Adaptor-specific weight initialisation.

        Must be called in the base PreTrainedModel's `_init_weights()` function. 

        Unlike the PEFT library -- which inits weights in the module init method or when an adaptor
        is added -- we must rely on the standard Trasnformers initialisation system so new adaptors
        are correctly initialised when a model is loaded with `from_pretrained()`. As a consequence
        we do not have access to the adaptors' configs which are passed during module init; we only
        have access to information stored in the module itself. If you need any special information
        to intialise the weights please save them as member variables and include them in the class
        `other_param_names` tuple so we can correctly track them!
        """
        raise NotImplementedError()

    @property
    def weight( self ) -> torch.Tensor:
        base_layer = self.get_base_layer()
        if hasattr( base_layer, 'qweight' ):
            # QuantLinear
            weight = base_layer.qweight
        else:
            # Other layers
            weight = base_layer.weight
        return weight # type: ignore

    @property
    def bias( self ) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return base_layer.bias # type: ignore

    @property
    def available_adaptors( self ) -> list[str]:
        """ Returns a sorted list of all available adaptor names """
        adaptors = set()
        for layer_name in self.adaptor_layer_names:
            module = getattr( self, layer_name )
            if not isinstance( module, ( nn.ModuleDict, nn.ParameterDict ) ):
                continue
            adaptors.update( set( module.keys() ) )
        return sorted( adaptors )

    def set_adaptors_trainable( self, adaptor_names: str | list[str] ):
        """ Sets adaptors to be trainable.
        The specified adaptors will have `requires_grad_` set to True, and all other will be set to False.
        """
        if isinstance( adaptor_names, str ):
            adaptor_names = [ adaptor_names ]
        for layer_name in self.adaptor_layer_names:
            module_dict = getattr( self, layer_name )
            for key, layer in module_dict.items():
                if key in adaptor_names:
                    layer.requires_grad_( True )
                else:
                    layer.requires_grad_( False )

    def __init__( self, base_layer: nn.Module, configs: dict[str, dict], **kwargs ):
        self.base_layer = base_layer

class ContextPeftAdaptorLora( nn.Module, ContextPeftAdaptorBase ):
    adaptor_layer_names = ( 'lora_A', 'lora_B' )
    other_param_names = ( 'r', 'lora_alpha', 'use_rslora', 'scaling' )
    module_types = ( nn.Linear, )

    def __init__( self, base_layer: nn.Module, configs: dict[str, dict], **kwargs ):
        super().__init__()
        ContextPeftAdaptorBase.__init__( self, base_layer, configs, **kwargs )

        # Helpers for in/out dim of base layer
        self.in_features = self.base_layer.in_features
        self.out_features = self.base_layer.out_features

        assert isinstance( self.in_features, int )
        assert isinstance( self.out_features, int )

        # Set rank, alpha and rslora hyperparameters
        self.r = { k: v['r'] for k, v in configs.items() }
        self.lora_alpha = { k: v['lora_alpha'] for k, v in configs.items() }
        self.use_rslora = { k: v['use_rslora'] for k, v in configs.items() }

        # Compute and store scaling or rslora scaling parameters
        scaling = { k: self.lora_alpha[k] / self.r[k] for k in configs }
        rs_scaling = { k: self.lora_alpha[k] / self.r[k] ** 0.5 for k in configs }
        self.scaling = { k: rs_scaling[k] if self.use_rslora[k] else scaling[k] for k in configs }

        # Compute if bias is enabled based on bool or 'auto' detection
        bias = { k: self.base_layer.bias is not None if v['use_bias'] == 'auto' else v['use_bias'] for k, v in configs.items() }

        # Create all A and B low rank matricies
        self.lora_A = nn.ModuleDict( { k: nn.Linear( self.in_features, self.r[k], False ) for k in configs } )
        self.lora_B = nn.ModuleDict( { k: nn.Linear( self.r[k], self.out_features, bias[k] ) for k in configs } )

    def init_adaptor_weights( self ):
        for a in self.lora_A.values():
            nn.init.kaiming_uniform_( a.weight, a=5 ** 0.5 )
        for b in self.lora_B.values():
            b.weight.data.zero_()
            if b.bias is not None:
                b.bias.data.zero_()

    def forward( self, x: torch.Tensor, *args, **kwargs ):

        # Get the adaptor mask
        adaptor_mask = kwargs.pop( 'adaptor_mask', {} )

        # Get a list of adaptors in this layer
        lora_A_keys = self.lora_A.keys()

        # Compute the original result and dtype
        result = self.base_layer( x, *args, **kwargs )
        result_dtype = result.dtype

        # Iterate over all adaptors in the adaptor mask
        for name, mask in adaptor_mask.items():
            # Check if that adaptor actuall exists
            if name in lora_A_keys:
                # Get A, B and scaling
                lora_A = self.lora_A[name]
                lora_B = self.lora_B[name]
                scaling = self.scaling[name]

                # Cast input to correct dtype # TODO: autocast logic?
                x = x.to( dtype=lora_A.weight.dtype )

                # Compute delta with scaling and masking
                delta = lora_B( lora_A( x ) ) * scaling * mask

                # Add delta to result, cast back to correct dtype
                result = result + delta.to( dtype=result_dtype )

        # Return updated result
        return result

class ContextPeftAdaptorIA3( nn.Module, ContextPeftAdaptorBase ):
    adaptor_layer_names = ( 'ia3_l', )
    other_param_names = tuple()
    module_types = ( nn.Linear, )

    def __init__( self, base_layer: nn.Module, configs: dict[str, dict], is_feedforward: bool, **kwargs ):
        super().__init__()
        ContextPeftAdaptorBase.__init__( self, base_layer, configs, **kwargs )

        self.is_feedforward = is_feedforward
        dim = self.base_layer.in_features if is_feedforward else self.base_layer.out_features
        assert isinstance( dim, int )

        self.ia3_l = nn.ParameterDict( { k: nn.Parameter( torch.empty( dim ) ) for k in configs } )

    def init_adaptor_weights( self ):
        for p in self.ia3_l.values():
            p.data.fill_( 1.0 )

    def forward( self, x: torch.Tensor, *args, **kwargs ):

        # Get the adaptor mask
        adaptor_mask = kwargs.pop( 'adaptor_mask', {} )

        # Get a list of adaptors in this layer
        ia3_l_keys = self.ia3_l.keys()

        # Create accumulator for IA3 gates
        ia3_scaling = 1

        # Iterate over all adaptors in the adaptor mask
        for name, mask in adaptor_mask.items():
            # Check if that adaptor actuall exists
            if name in ia3_l_keys:
                # Get gate
                ia3_l = self.ia3_l[name]

                # Update scaling gate based on mask
                ia3_scaling = torch.where( mask, ia3_l * ia3_scaling, ia3_scaling )

        if self.is_feedforward:
            # If feedforward apply gate to input
            interm = ( x * ia3_scaling ).to( dtype=x.dtype )
            result = self.base_layer( interm, *args, **kwargs )
        else:
            # Otherwise compute result and then gate
            result = self.base_layer( x, *args, **kwargs )
            result = ( result * ia3_scaling ).to( dtype=result.dtype )

        # Return updated result
        return result

class ContextPeftAdaptorBitFit( nn.Module, ContextPeftAdaptorBase ):
    adaptor_layer_names = ( 'biases', )
    other_param_names = tuple()
    module_types = ( nn.Linear, )

    def __init__( self, base_layer: nn.Module, configs: dict[str, dict], **kwargs ):
        super().__init__()
        ContextPeftAdaptorBase.__init__( self, base_layer, configs, **kwargs )

        dim = self.base_layer.out_features
        assert isinstance( dim, int )

        self.biases = nn.ParameterDict( { k: nn.Parameter( torch.empty( dim ) ) for k, v in configs.items() if v[ 'force_bias' ] or self.base_layer.bias is not None } )

    def init_adaptor_weights( self ):
        for p in self.biases.values():
            p.data.zero_()

    def forward( self, x: torch.Tensor, *args, **kwargs ):

        # Get the adaptor mask
        adaptor_mask = kwargs.pop( 'adaptor_mask', {} )

        # Get a list of adaptors in this layer
        bias_keys = self.biases.keys()

        # Compute the original result and dtype
        result = self.base_layer( x, *args, **kwargs )
        result_dtype = result.dtype

        # Iterate over all adaptors in the adaptor mask
        for name, mask in adaptor_mask.items():
            # Check if that adaptor actuall exists
            if name in bias_keys:
                # Get bias
                bias = self.biases[name]

                # Compute delta with scaling and masking
                delta = bias * mask

                # Add delta to result, cast back to correct dtype
                result = result + delta.to( dtype=result_dtype )

        # Return updated result
        return result



class ContextPeftWrapperBase( ABC ):
    adaptor_class: type[ContextPeftAdaptorBase]

    @classmethod
    @abstractmethod
    def get_layers( cls, model: nn.Module, adaptors: dict ) -> list[tuple[dict[str,list[str]], dict]]:
        """ Get layers to transform.

        Args:
            model (torch.nn.Module): The base model to be adapted.
            adaptors (dict): Dictionary of all PEFT adaptors.

        Returns:
            A list of [dict, dict] tuples:
            - The first dict maps layer names to adaptor names.
            - The second dict provides additional kwargs to specialise the adaptor type.
        """

    @classmethod
    def set_layers( cls, model: nn.Module, adaptors: dict, layer_map: list[tuple[dict[str,list[str]], dict]] ):
        # Iterate over layer map getting the layer group and it's kwargs
        for group, kwargs in layer_map:

            # Iterate over group getting layer name and adaptor name list
            for layer_name, adaptor_names in group.items():

                # Get name of parent layer and child attribute name
                parent_name = '.'.join( layer_name.split( '.' )[ : -1 ] )
                child_name = layer_name.split( '.' )[-1]

                # Get parent module by name and child by attr name
                parent_module = model.get_submodule( parent_name )
                child_module = parent_module.get_submodule( child_name )

                # Get specific adaptor configs needed for this layer
                configs = { k: adaptors[k] for k in adaptor_names }

                # Construct adaptor wrapper to replace layer
                new_module = cls.adaptor_class( child_module, configs, **kwargs )

                # Replace child with adaptor
                setattr( parent_module, child_name, new_module )

    @classmethod
    def replace_layers( cls, model: nn.Module, adaptors: dict ):
        # Get layer map
        layer_map = cls.get_layers( model, adaptors )

        # Set layers
        cls.set_layers( model, adaptors, layer_map )

class ContextPeftWrapperGeneric( ContextPeftWrapperBase ):
    @classmethod
    def get_layers( cls, model: nn.Module, adaptors: dict ) -> list[tuple[dict[str,list[str]], dict]]:
        # Just one layer map needed as no special kwargs used
        layer_map = {}

        # Iterate over all modules by name
        for layer_name, module in model.named_modules():

            # List of adaptors we'll enable for this layer
            active_adaptors = []

            # Iterate over all adaptor names and configs
            for adaptor_name, adaptor_config in adaptors.items():

                # Get target modules as a list
                target_modules = adaptor_config[ 'target_modules' ]
                if isinstance( target_modules, str ):
                    target_modules = [ target_modules ]

                # Get excluded modules as list
                exclude_modules = adaptor_config[ 'exclude_modules' ] or []
                if isinstance( exclude_modules, str ):
                    exclude_modules = [ exclude_modules ]

                # If the layer name matches target modules but does NOT match exluded modules add to active adaptors
                if any( t in layer_name for t in target_modules ) and not any( e in layer_name for e in exclude_modules ):
                    active_adaptors.append( adaptor_name )

            # If we have at least one active adaptor
            if active_adaptors:

                # Check the layer is a wrappable type
                if not isinstance( module, cls.adaptor_class.module_types ):
                    raise ValueError( f'Tried to wrap layer {layer_name} with invalid type {module.__class__.__name__}!' )

                # Add layer and adaptor names to the map
                layer_map[ layer_name ] = active_adaptors

        # Return layer map and an empty kwargs dict
        return [ ( layer_map, {} ) ]

class ContextPeftWrapperLora( ContextPeftWrapperGeneric ):
    adaptor_class = ContextPeftAdaptorLora

class ContextPeftWrapperBitFit( ContextPeftWrapperGeneric ):
    adaptor_class = ContextPeftAdaptorBitFit

class ContextPeftWrapperIA3( ContextPeftWrapperBase ):
    adaptor_class = ContextPeftAdaptorIA3

    @classmethod
    def get_layers( cls, model: nn.Module, adaptors: dict ) -> list[tuple[dict[str,list[str]], dict]]:
        # IA3 is special, we need layer maps for attn projs and for ffn down projs
        proj_layer_map = {}
        ff_layer_map = {}

        # Iterate over all modules by name
        for layer_name, module in model.named_modules():

            # List of proj/ffn adaptors we'll enable for this layer
            proj_active_adaptors = []
            ff_active_adaptors = []

            # Iterate over all adaptor names and configs
            for adaptor_name, adaptor_config in adaptors.items():

                # Get target modules as a list
                target_modules = adaptor_config[ 'target_modules' ]
                if isinstance( target_modules, str ):
                    target_modules = [ target_modules ]

                # Get excluded modules as a list
                exclude_modules = adaptor_config[ 'exclude_modules' ] or []
                if isinstance( exclude_modules, str ):
                    exclude_modules = [ exclude_modules ]

                # Get FFN modules as a list
                feedforward_modules = adaptor_config[ 'feedforward_modules' ] or []
                if isinstance( feedforward_modules, str ):
                    feedforward_modules = [ feedforward_modules ]

                # If the layer name matches target modules but does NOT match exluded modules...
                if any( t in layer_name for t in target_modules ) and not any( e in layer_name for e in exclude_modules ):
                    # Add to proj if NOT a match to FFN modules
                    if not any( f in layer_name for f in feedforward_modules ):
                        proj_active_adaptors.append( adaptor_name )
                    # Add to proj if IS a match to FFN modules
                    else:
                        ff_active_adaptors.append( adaptor_name )

            # Check the layer is a wrappable type
            if proj_active_adaptors or ff_active_adaptors:
                if not isinstance( module, cls.adaptor_class.module_types ):
                    raise ValueError( f'Tried to wrap layer {layer_name} with invalid type {module.__class__.__name__}!' )

            # Add layer and adaptor names to proj map
            if proj_active_adaptors:
                proj_layer_map[ layer_name ] = proj_active_adaptors

            # Add layer and adaptor names to FFN map
            if ff_active_adaptors:
                ff_layer_map[ layer_name ] = ff_active_adaptors

            # If layer is somehow in both, there's something wrong!
            if proj_active_adaptors and ff_active_adaptors:
                raise ValueError( f'Layer {layer_name} marked as feedforward=True and feedforward=False by different adaptors!' )

        # Return the layer maps with the is_feedforward kwarg set!
        return [
            ( proj_layer_map, { 'is_feedforward': False } ),
            ( ff_layer_map, { 'is_feedforward': True } ),
        ]

CONTEXT_PEFT_WRAPPER_MAPPING = {
    'lora': ContextPeftWrapperLora,
    'ia3': ContextPeftWrapperIA3,
    'bitfit': ContextPeftWrapperBitFit,
}

class ContextPeftConnector( nn.Module ):
    def __init__( self, in_features: int, out_features: int, connector_activation: str, connector_bias: bool ):
        super().__init__()

        self.in_proj = nn.Linear( in_features, out_features, bias=connector_bias )
        self.out_proj = nn.Linear( out_features, out_features, bias=connector_bias )
        self.act_fn = ACT2FN[connector_activation]

    def forward( self, x ):
        x = self.in_proj( x )
        x = self.act_fn( x )
        x = self.out_proj( x )
        return x

class ContextPeftPreTrainedModel( PreTrainedModel ):
    config_class = ContextPeftConfig
    base_model_prefix = 'text_model'

    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True

    def _init_weights( self, module ):
        std = self.config.get_text_config().initializer_range

        if isinstance( module, ( nn.Linear, nn.Conv2d ) ):
            module.weight.data.normal_( mean=0.0, std=std )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance( module, nn.Embedding ):
            module.weight.data.normal_( mean=0.0, std=std )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance( module, ContextPeftAdaptorBase ):
            module.init_adaptor_weights()

# Special pre forward hook for adaptor mask injection!
def _adaptor_mask_pre_forward_hook( target, args, kwargs, adaptor_mask ):
    kwargs[ 'adaptor_mask' ] = adaptor_mask
    return args, kwargs

class ContextPeftForConditionalGeneration( ContextPeftPreTrainedModel, GenerationMixin ):
    def __init__( self, config: ContextPeftConfig, load_from_hub=False, **kwargs ):
        super().__init__( config, **kwargs )

        # Set local versions because self.config.* can cause graph breaks
        self.image_pad_token_id = config.image_pad_token_id
        self.adaptor_map = config.adaptor_map

        # Not needed, but makes code cleaner
        vision_config = config.vision_config
        text_config = config.text_config

        # Get correct dtype for if we're training either backbone
        vision_dtype = torch.float32 if config.vision_trainable else vision_config.torch_dtype
        text_dtype = torch.float32 if config.text_trainable else text_config.torch_dtype

        if not load_from_hub:
            # If not loading from hub (e.g. inference or adding extra adaptors)
            self.vision_model: PreTrainedModel = AutoModel.from_config( vision_config, torch_dtype=vision_dtype, **kwargs ).requires_grad_( config.vision_trainable )
            self.text_model: PreTrainedModel = AutoModelForCausalLM.from_config( text_config, torch_dtype=text_dtype, **kwargs ).requires_grad_( config.text_trainable)
        else:
            # Load from hub when making new Context-PEFT model
            self.vision_model: PreTrainedModel = AutoModel.from_pretrained( vision_config._name_or_path, config=vision_config, torch_dtype=vision_dtype, **kwargs ).requires_grad_( config.vision_trainable )
            self.text_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained( text_config._name_or_path, config=text_config, torch_dtype=text_dtype, **kwargs ).requires_grad_( config.text_trainable )

        if config.peft_type is not None:
            # Set flag to enable adaptors
            self.peft_enabled = True

            # Wrap all layers with adaptors using the relevant ContextPeftTuner* class
            CONTEXT_PEFT_WRAPPER_MAPPING[ config.peft_type ].replace_layers( self.text_model, config.adaptors )

            # Set only active adaptors to be trainable
            self.set_adaptors_trainable( True )
        else:
            # No adaptors present, set flag to false
            self.peft_enabled = False

        # Module dict for all multi-modal connectors (only text for now)
        self.connector = nn.ModuleDict( {} )
        self.connector[ 'image' ] = ContextPeftConnector(
            config.vision_dim,
            config.text_dim,
            config.connector_activation,
            config.connector_bias
        )

        # If the text model has tied weights we must add their keys
        if self.text_model._tied_weights_keys is not None:
            self._tied_weights_keys = [ f'text_model.{k}' for k in self.text_model._tied_weights_keys ] # type: ignore
        
        # Set support flags
        self._supports_flash_attn_2 = self.text_model._supports_flash_attn_2
        self._supports_sdpa = self.text_model._supports_sdpa
        self._supports_flex_attn = self.text_model._supports_flex_attn
        self._supports_cache_class = self.text_model._supports_cache_class
        self._supports_static_cache = self.text_model._supports_static_cache
        self._supports_quantized_cache = self.text_model._supports_quantized_cache

        # Run standard HF post init
        self.post_init()

    def set_adaptors_trainable( self, adaptors: bool | list[str] = True ):
        """ Sets adaptors to be trainable.
        The specified adaptors will have `requires_grad_` set to True, and all other will be set to False.

        Args:
            adaptors (bool | list[str], optional): The list of adaptor names to set trainable.
                When True sets only adaptors in `config.active_adaptors` to be trainable.
                When False sets no adaptors to be trainable. Defaults to True.
        """
        if isinstance( adaptors, bool ):
            adaptors = self.config.active_adaptors if adaptors else []
            assert isinstance( adaptors, list )

        for module in self.text_model.modules():
            if isinstance( module, ContextPeftAdaptorBase ):
                module.set_adaptors_trainable( adaptors )

    def get_adaptor_mask(
        self,
        input_ids: torch.LongTensor,
        skip_unused_adaptors=False
    ) -> dict[str, torch.Tensor]:
        """ Computes the adaptor mask for the current input ids.

        NOTE: setting `skip_unused_adaptors=True` can improve inference latency but is incompatible with `torch.compile`!

        Args:
            input_ids (torch.LongTensor): Input ids of the current sequence.
            skip_unused_adaptors (bool, optional): Skips adaptors inactive for the entire batch. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: _description_
        """
        
        # Dict which maps adaptor name -> mask
        adaptor_mask: dict[str, torch.Tensor] = {}

        # Map context -> mask
        context_map = {
            'text': input_ids != self.image_pad_token_id,
            'image': input_ids == self.image_pad_token_id,
        }

        # Iterate over all adaptor names
        for adaptor_name, contexts in self.adaptor_map.items():
            # Create empty mask to accumulate masks into
            mask = torch.zeros_like( input_ids, dtype=torch.bool )

            # Iterate over adaptor's contexts and update mask
            for c in contexts:
                mask += context_map[c]

            # Write mask into dict
            adaptor_mask[ adaptor_name ] = mask.unsqueeze( -1 )

        # During inference we may want to skip unused adaptors
        if skip_unused_adaptors:
            # Retain only adaptor masks that are used at least once within this batch
            adaptor_mask = { k: v for k, v in adaptor_mask.items() if torch.any( v ) }

        return adaptor_mask

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        others_embeds: torch.Tensor | None,
        others_pad_token_id: int
    ) -> torch.Tensor:
        """ Merges embeddings from another modality with text embeddings.

        Returns `input_embeds` where corresponding positions of `input_ids` equal to
        `others_pad_token_id` have been replaced by the contents of `others_embeds`.

        Args:
            input_ids (torch.LongTensor): Sequence of input token ids with shape [B, S].
            inputs_embeds (torch.Tensor): Sequence of input embeddings with shape [B, S, D]
            others_embeds (torch.Tensor | None): Sequence of embeddings from another modality with shape [*, *, D]
            others_pad_token_id (int): Padding token id for the other modality. 

        Returns:
            torch.Tensor: The merged embeddings, or unchanged input_embeds if others_embeds is None.
        """
        # If other modality not present skip merge
        if others_embeds is None:
            return inputs_embeds

        # Get dimensions of inputs and others
        i_batch, i_seq, i_dim = inputs_embeds.shape
        o_batch, o_seq, o_dim = others_embeds.shape

        # Flatten batch and sequence dimensions
        input_ids = input_ids.reshape( i_batch * i_seq ) # type: ignore
        inputs_embeds = inputs_embeds.reshape( [ i_batch * i_seq, i_dim ] )
        others_embeds = others_embeds.reshape( [ o_batch * o_seq, o_dim ] ).to( dtype=inputs_embeds.dtype )

        # Get a token mask for other modality
        token_mask = input_ids == others_pad_token_id

        # Update inputs by other modality (no need to check for missmatch, will throw error!)
        inputs_embeds = inputs_embeds.masked_scatter( token_mask.unsqueeze( -1 ), others_embeds )

        # Reshape back to [B, S, D]
        return inputs_embeds.reshape( i_batch, i_seq, i_dim )

    @contextmanager
    def _enable_forward_hooks( self, *args, **kwargs ):
        adaptor_mask = kwargs.pop( 'adaptor_mask', None )

        # Skip if PEFT disabled!
        if adaptor_mask is None:
            yield
            return

        # We must tack hooks to remove after forward!
        hook_handles = []

        # Iterate over all modules and apply hook to adaptors
        for module in self.text_model.modules():
            if isinstance( module, ContextPeftAdaptorBase ):
                pre_forward = partial( _adaptor_mask_pre_forward_hook, adaptor_mask=adaptor_mask )
                handle = module.register_forward_pre_hook( pre_forward, with_kwargs=True )
                hook_handles.append( handle )

        # Yield to wrapped code
        yield

        # Finally, remove all hooks
        for handle in hook_handles:
            handle.remove()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # position_ids: torch.LongTensor | None = None,
        # past_key_values: list[torch.FloatTensor] | Cache | None = None,
        # cache_position: torch.LongTensor | None = None,
        # inputs_embeds: torch.FloatTensor | None = None,
        # use_cache: bool | None = None,
        # return_dict: bool | None = None,
        # output_attentions: bool | None = None,
        # output_hidden_states: bool | None = None,
        # logits_to_keep: int | torch.Tensor = 0,
        skip_unused_adaptors=False,
        **kwargs
    ):
        # Todo: new logic for input_ids AND inputs_embeds?
        if input_ids is None:
            raise ValueError( 'input_ids must always be passed!' )

        # Get text embeddings
        inputs_embeds = self.text_model.get_input_embeddings()( input_ids )

        # Get image embeddings and project
        if pixel_values is not None:
            images_embeds = self.vision_model( pixel_values ).last_hidden_state
            images_embeds = self.connector[ 'image' ]( images_embeds )
        else:
            images_embeds = None

        # Merge images into input embeddings
        inputs_embeds = self.inputs_merger( input_ids, inputs_embeds, images_embeds, self.image_pad_token_id )

        # Get adaptor mask if peft is enabled, otherwise skip as no adaptors present
        adaptor_mask = self.get_adaptor_mask( input_ids, skip_unused_adaptors ) if self.peft_enabled else None

        # Add adaptor mask forward hooks to all adaptor layers
        with self._enable_forward_hooks( adaptor_mask=adaptor_mask ):
            # Forward pass text model with merged embeddings and additional kwargs
            return self.text_model( inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs )

    def tie_weights(self):
        # Ensure text model performs weight tying
        return self.text_model.tie_weights()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        logits_to_keep=None,
        use_cache=True,
        skip_unused_adaptors=None,
        **kwargs,
    ):
        # Hijack text model's prepare function
        model_inputs = self.text_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # TODO: improve this? Copied from Gemma3
        # Only pass pixel_values during pre-fill
        if cache_position[0] == 0: # type: ignore
            model_inputs[ 'pixel_values' ] = pixel_values

        # Include skip_unused_adaptors if specified
        if skip_unused_adaptors is not None:
            model_inputs[ 'skip_unused_adaptors' ] = skip_unused_adaptors
        
        return model_inputs
    
    def _supports_logits_to_keep( self ):
        # Cheeky hack to get logits_to_keep support from text model
        return self.text_model._supports_logits_to_keep() # pylint: disable=W0212