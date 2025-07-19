from dataclasses import dataclass, field
from typing import Optional, Literal, TypeAlias, get_args

ADAPTOR_METHODS: TypeAlias = Literal['connector', 'fullft', 'lora', 'bitfit', 'ia3']
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

    dataset: str = field( default='coco' )
    evaluation_datasets: list[str] = field( default_factory=list )
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
    optimizer: str = field( default='adamw' )
    optimizer_percentile_clipping: Optional[int] = field( default=None )

    trainable_embeddings: Optional[bool] = field( default=False )

    connector_dropout: float = field( default=0.0 )
    connector_bias: bool = field( default=False )

    adaptor_method: ADAPTOR_METHODS = field( default='connector' )
    adaptor_defaults: Optional[dict] = field( default=None )
    adaptor_additions: Optional[dict] = field( default=None )
    adaptor_dropout: Optional[float | dict[str, float]] = field( default=None )

    trainable_adaptors: Optional[list[str] | bool] = field( default=None )

    train_compile_mode: Optional[COMPILE_MODES] = field( default=None )
    validation_compile_mode: Optional[COMPILE_MODES] = field( default=None )

    meta: Optional[dict] = field( default=None )

    seed_offset: int = field( default=0 )

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

        # Create new peft_type and text_trainable flags
        self.peft_type = None if self.adaptor_method in [ 'connector', 'fullft' ] else self.adaptor_method
        self.text_trainable = self.adaptor_method == 'fullft'

        # Ensure adaptor_context is set iff adaptor_method is a PEFT type
        if self.peft_type is None and self.adaptor_defaults is not None:
            raise ValueError( f'Adaptor method {self.adaptor_method} requires adaptor_defaults=None' )
        if self.peft_type is None and self.adaptor_additions is not None:
            raise ValueError( f'Adaptor method {self.adaptor_method} requires adaptor_additions=None' )
        if self.peft_type is None and self.adaptor_dropout is not None:
            raise ValueError( f'Adaptor method {self.adaptor_method} requires adaptor_dropout=None' )

        # Ensure adaptor_decay not set if adaptor_method is not a PEFT type
        if self.peft_type is None and self.adaptor_decay is not False:
            raise ValueError( 'adaptor_decay must be False when no adaptors are active!' )

        if self.trainable_embeddings is None:
            self.trainable_embeddings = self.text_trainable

        if self.peft_type is None and self.trainable_adaptors is not None:
            raise ValueError( f'Cannot set any trainable adaptors with adaptor method {self.adaptor_method}' )
        if self.peft_type is not None and self.trainable_adaptors is None:
            self.trainable_adaptors = True