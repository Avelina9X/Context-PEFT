import math

def _get_min_lr( lr: float, **kwargs ):
    min_lr = kwargs.get( 'min_lr', None )
    min_lr_ratio = kwargs.get( 'min_lr_ratio', None )

    if not( ( min_lr is None ) ^ ( min_lr_ratio is None ) ):
        raise ValueError( 'Must specify exactly one of min_lr or min_lr_ratio!' )

    if min_lr is None:
        assert isinstance( min_lr_ratio, float )
        min_lr = min_lr_ratio * lr
    else:
        assert isinstance( min_lr, float )

    if min_lr > lr:
        raise ValueError( 'Minimum learning rate is greater than maximum learning rate!' )

    return min_lr

class BaseSchedule:
    def __init__( self, warmup_steps: int, total_training_steps: int, lr: float, **kwargs ):
        self.warmup_steps = max( 1, warmup_steps )
        self.total_training_steps = total_training_steps
        self.max_lr = lr

    def get_lr( self, current_step: int ) -> float:
        warmup = min( 1.0, current_step / self.warmup_steps )
        return self.max_lr * warmup

class ConstantSchedule( BaseSchedule ):
    def __init__( self, warmup_steps: int, total_training_steps: int, lr: float, **kwargs ):
        super().__init__( warmup_steps, total_training_steps, lr )
        if kwargs:
            raise ValueError( 'Cannot specify kwargs for ConstantSchedule!' )

class LinearSchedule( BaseSchedule ):
    def __init__( self, warmup_steps: int, total_training_steps: int, lr: float, **kwargs ):
        super().__init__( warmup_steps, total_training_steps, lr )
        self.min_lr = _get_min_lr( lr, **kwargs )

    def get_lr( self, current_step: int ):
        max_lr = super().get_lr( current_step )

        cooldown_end = self.total_training_steps - self.warmup_steps
        cooldown_step = min( max( 0, current_step - self.warmup_steps ), cooldown_end )
        cooldown_t = cooldown_step / cooldown_end

        return cooldown_t * self.min_lr + ( 1 - cooldown_t ) * max_lr

class CosineSchedule( BaseSchedule ):
    def __init__( self, warmup_steps: int, total_training_steps: int, lr: float, **kwargs ):
        super().__init__( warmup_steps, total_training_steps, lr )
        self.min_lr = _get_min_lr( lr, **kwargs )

    def get_lr( self, current_step: int ):
        max_lr = super().get_lr( current_step )

        cooldown_end = self.total_training_steps - self.warmup_steps
        cooldown_step = min( max( 0, current_step - self.warmup_steps ), cooldown_end )
        cooldown_t = cooldown_step / cooldown_end
        cooldown_t = math.cos( cooldown_t * math.pi ) / 2 + 0.5
        cooldown_t = 1.0 - cooldown_t

        return cooldown_t * self.min_lr + ( 1 - cooldown_t ) * max_lr

SCHEDULE_MAP: dict[str, type[BaseSchedule]] = {
    'constant': ConstantSchedule,
    'linear': LinearSchedule,
    'cosine': CosineSchedule,
}