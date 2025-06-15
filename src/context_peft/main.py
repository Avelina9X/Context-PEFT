import argparse
import yaml

from trainer import Trainer, TrainerConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--config', required=True, type=str )
    parser.add_argument( '--seed_offset', required=False, type=int )
    args = parser.parse_args()

    with open( args.config, 'r', encoding='utf-8' ) as f:
        config = yaml.safe_load( f )
        assert isinstance( config, dict )
    
    trainer_config = TrainerConfig(
        **config
    )

    if args.seed_offset is not None:
        trainer_config.seed_offset = args.seed_offset

    trainer = Trainer( trainer_config )
    trainer.train()