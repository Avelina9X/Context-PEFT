import argparse
import yaml

from trainer import Trainer, TrainerConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--config', required=True, type=str )
    args = parser.parse_args()

    with open( args.config, 'r', encoding='utf-8' ) as f:
        config = yaml.safe_load( f )
        assert isinstance( config, dict )
    
    trainer_config = TrainerConfig(
        **config
    )

    trainer = Trainer( trainer_config )
    trainer.train()