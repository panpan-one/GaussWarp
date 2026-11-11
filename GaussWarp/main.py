import os
import random
import datetime
import torch
import numpy as np
from exp.exp_main import Exp_Main
from utils.globals import logger, accelerator
from utils.configs import configs

def seed_training(seed):
    configs.seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if configs.dataset_name == 'P12':
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def main():
    _seed_override = os.environ.get('FIX_SEED_LIST', '').strip()
    fix_seed_list = [int(s) for s in _seed_override.split(',') if s.strip()] if _seed_override else list(range(configs.seed, configs.seed + configs.itr))
    configs.use_gpu = True if torch.cuda.is_available() and configs.use_gpu else False
    Exp = Exp_Main
    if configs.sweep:
        logger.info('>>>>>>> sweeping start <<<<<<<')
        subfolder = datetime.datetime.now().strftime('%Y_%m%d_%H%M')
        configs.subfolder_train = subfolder
        configs.wandb = 1
        logger.debug('wandb=1: Weight & Bias logging is automatically enabled')
        configs.itr = 1
        configs.itr_i = 0
        logger.debug('itr=1: training iteration is automatically overwritten to 1')
        seed_training(fix_seed_list[configs.itr_i])
        exp = Exp(configs)
        exp.train()
        if configs.test_after_train:
            exp.test()
    elif configs.is_training:
        subfolder = datetime.datetime.now().strftime('%Y_%m%d_%H%M')
        configs.subfolder_train = subfolder
        for i in range(configs.itr):
            configs.itr_i = i
            seed_training(fix_seed_list[i])
            exp = Exp(configs)
            exp.train()
            torch.cuda.empty_cache()
        if configs.test_after_train:
            exp.test()
    else:
        exp = Exp(configs)
        exp.test()
        torch.cuda.empty_cache()
if __name__ == '__main__':
    try:
        if not configs.sweep:
            main()
        else:
            import wandb
            sweep_configuration = {'method': 'grid', 'metric': {'goal': 'minimize', 'name': 'loss_test'}, 'parameters': {'learning_rate': {'values': [0.01, 0.001, 0.0001, 1e-05]}, 'batch_size': {'values': [16, 32, 64, 128]}}}
            temp_file_path = 'storage/tmp.txt'
            if accelerator.is_main_process:
                sweep_id = wandb.sweep(sweep=sweep_configuration, project='YOUR_PROJECT_NAME')
                with open(temp_file_path, mode='w', encoding='utf-8') as f:
                    f.write(sweep_id)
            accelerator.wait_for_everyone()
            sweep_id = None
            with open(temp_file_path, mode='r', encoding='utf-8') as f:
                sweep_id = f.readline()
            wandb.agent(sweep_id, function=main, project='YOUR_PROJECT_NAME', count=16)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print('\nProcess interrupted...')
    except Exception as e:
        if accelerator.is_main_process:
            logger.exception(f'{e}', stack_info=True)
            exit(1)
