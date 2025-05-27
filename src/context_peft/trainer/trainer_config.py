from dataclasses import dataclass, field
from typing import Optional, Literal, TypeAlias, get_args

ADAPTOR_METHODS: TypeAlias = Literal['connector', 'fullft', 'lora', 'bitfit', 'ia3']
ADAPTOR_CONTEXTS: TypeAlias = Literal['image', 'text', 'both', 'shared']
COMPILE_MODES: TypeAlias = Literal['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']

@dataclass
class TrainerConfig:

    run_name: str
    output_dir: str | None
    
    wandb_group: str
    wandb_mode: Literal['online', 'offline', 'disabled']
    wandb_tags: list[str] = field( default_factory=list )

    text_model_name: Optional[str] = field( default=None )
    vision_model_name: Optional[str] = field( default=None )
    cpeft_model_path: Optional[str] = field( default=None )

    stage: Literal['stage1', 'stage2'] = field( default='stage1' )

    batch_size: int = field( default=32 )
    micro_batch_size: int = field( default=-1 )

    dataset: Literal['coco'] = field( default='coco' )
    sequence_length: int = field( default=-1 )
    pad_to_multiple: int = field( default=32 )
    dataset_train_workers: int = field( default=1 )
    dataset_validation_worker: bool = field( default=False )

    num_train_epochs: float = field( default=1.0 )
    logging_steps: int = field( default=64 )
    validation_interval: int = field( default=8 )
    evaluation_interval: int = field( default=64 )
    
    learning_rate: float = field( default=1e-4 )
    learning_rate_schedule: str = field( default='constant' )
    learning_rate_schedule_kwargs: dict = field( default_factory=dict )
    warmup_steps: int | float = field( default=64 )

    weight_decay: float = field( default=0.1 )
    adaptor_decay: bool | float = field( default=False )
    adam_beta1: float = field( default=0.9 )
    adam_beta2: float = field( default=0.999 )
    adam_eps: float = field( default=1e-8 )
    max_grad_norm: float = field( default=1.0 )

    connector_dropout: float = field( default=0.0 )
    connector_bias: bool = field( default=False )

    adaptor_method: ADAPTOR_METHODS = field( default='connector' )
    adaptor_context: Optional[ADAPTOR_CONTEXTS] = field( default=None )
    lora_rank: Optional[int] = field( default=None )
    adaptor_dropout: float = field( default=0.0 )

    train_compile_mode: Optional[COMPILE_MODES] = field( default=None )
    validation_compile_mode: Optional[COMPILE_MODES] = field( default=None )

    def __post_init__( self ):
        self._validate_stage()
        self._validate_batch()
        self._validate_adaptors()

    def _validate_stage( self ):
        if self.stage == 'stage1':
            if self.text_model_name is None:
                raise ValueError( 'text_model_name must be specified in stage 1!' )
            if self.vision_model_name is None:
                raise ValueError( 'vision_model_name must be specified in stage 1!' )
            if self.cpeft_model_path is not None:
                raise ValueError( 'cpeft_model_path must not be specified in stage 1!' )
        elif self.stage == 'stage2':
            if self.text_model_name is None:
                raise ValueError( 'text_model_name must be carried over from stage 1!' )
            if self.vision_model_name is None:
                raise ValueError( 'vision_model_name must be carried over from stage 1!' )
            if self.cpeft_model_path is None:
                raise ValueError( 'cpeft_model_path must be specified in stage 2!' )
        else:
            raise ValueError( f'Unknown stage "{self.stage}"' )

    def _validate_batch( self ):
        if self.micro_batch_size == -1:
            self.micro_batch_size = self.batch_size
        else:
            if self.batch_size % self.micro_batch_size != 0:
                raise ValueError( 'micro_batch_size must be an integer division of batch_size!' )

    def _validate_adaptors( self ):
        # Ensure adaptor_method and adaptor_context are valid options
        if self.adaptor_method not in get_args( ADAPTOR_METHODS ):
            raise ValueError( f'Invalid adaptor_method {self.adaptor_method}' )
        if self.adaptor_context is not None and self.adaptor_context not in get_args( ADAPTOR_CONTEXTS ):
            raise ValueError( f'Invalid adaptor_context {self.adaptor_context}' )

        # Create new peft_type and text_trainable flags
        self.peft_type = None if self.adaptor_method in [ 'connector', 'fullft' ] else self.adaptor_method
        self.text_trainable = self.adaptor_method == 'fullft'

        # Ensure adaptor_context is set iff adaptor_method is a PEFT type
        if self.peft_type is None and self.adaptor_context is not None:
            raise ValueError( f'Adaptor method {self.adaptor_method} requires adaptor_context=None' )
        if self.peft_type is not None and self.adaptor_context is None:
            raise ValueError( f'Adaptor method {self.adaptor_method} requires adaptor_context to be set!' )

        # Ensure adaptor_decay not set if adaptor_method is not a PEFT type
        if self.peft_type is None and self.adaptor_decay is not False:
            raise ValueError( 'adaptor_decay must be False when no adaptors are active!' )

        # Ensure lora_rank is set iff adaptor_method is lora
        if self.adaptor_method == 'lora' and self.lora_rank is None:
            raise ValueError( 'Adaptor method lora requires lora_rank to be set!' )
        if self.adaptor_method != 'lora' and self.lora_rank is not None:
            raise ValueError( f'Cannot set lora_rank for adaptor method {self.adaptor_method}!' )
