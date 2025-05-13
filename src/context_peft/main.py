
import torch

from trainer.trainer import Trainer
from trainer.trainer_config import TrainerConfig

if __name__ == '__main__':
    torch._logging.set_logs(
        graph_breaks=True,
        recompiles=True,
    )

    vision_model_name = 'openai/clip-vit-base-patch32'

    seq_len = {
        'openai/clip-vit-base-patch32': 160,
        'openai/clip-vit-base-patch16': 320,
        'openai/clip-vit-large-patch14': 384,
        'openai/clip-vit-large-patch14-336': -1,
    }[ vision_model_name ]

    # torch._dynamo.config.compiled_autograd = False
    
    trainer_config = TrainerConfig(
        run_name='placeholder',
        output_dir='placeholder',
        
        text_model_name='Qwen/Qwen1.5-0.5B-Chat',
        vision_model_name=vision_model_name,

        stage='stage1',
        
        batch_size=64,
        micro_batch_size=-1,

        dataset='coco',
        sequence_length=seq_len,
        pad_to_multiple=32,
        dataset_train_workers=8,
        dataset_validation_worker=True,

        num_train_epochs=8.0,
        logging_steps=256 * 4,
        validation_interval=4 // 4,
        evaluation_interval=1600,

        learning_rate=1e-4,
        learning_rate_schedule='constant',
        learning_rate_schedule_kwargs={},
        warmup_steps=64,

        weight_decay=0.1,
        adaptor_decay=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        max_grad_norm=1.0,

        adaptor_method='connector',
        adaptor_context=None,
        lora_rank=None,

        train_compile_mode='default',
        validation_compile_mode='default',
    )

    trainer = Trainer( trainer_config )
    trainer.train()

    # for _ in tqdm.tqdm( trainer.get_train_dataloader() ):
    #     pass