import argparse
import yaml

from trainer import Trainer, TrainerConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--config', required=True, type=str )
    parser.add_argument( '--seed_offset', required=False, type=int )
    parser.add_argument( '--micro_batch_mult', required=False, type=int, default=1 )
    parser.add_argument( '--wandb_mode', required=False, type=str )
    args = parser.parse_args()

    with open( args.config, 'r', encoding='utf-8' ) as f:
        config = yaml.safe_load( f )
        assert isinstance( config, dict )

    config[ 'micro_batch_size' ] = min( config[ 'batch_size' ], config[ 'micro_batch_size' ] * args.micro_batch_mult )
    
    trainer_config = TrainerConfig(
        **config
    )

    if args.seed_offset is not None:
        trainer_config.seed_offset = args.seed_offset

    if args.wandb_mode is not None:
        trainer_config.wandb_mode = args.wandb_mode

    trainer = Trainer( trainer_config )
    trainer.train()