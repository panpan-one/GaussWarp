import importlib
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils.ExpConfigs import ExpConfigs

def data_provider(configs: ExpConfigs, flag: str, shuffle_flag: bool=None, drop_last: bool=None) -> tuple[Dataset, DataLoader]:
    dataset_module = importlib.import_module(f'data.data_provider.datasets.{configs.dataset_name}')
    Data = dataset_module.Data
    collate_fn = dataset_module.collate_fn
    if flag in ['test', 'test_all']:
        shuffle_flag = False
        drop_last = False
        batch_size = configs.batch_size
    elif flag == 'val':
        shuffle_flag = False if configs.train_val_loader_shuffle is None else bool(configs.train_val_loader_shuffle)
        drop_last = False
        batch_size = configs.batch_size
    else:
        shuffle_flag = True if configs.train_val_loader_shuffle is None else bool(configs.train_val_loader_shuffle)
        drop_last = True if configs.train_val_loader_drop_last is None else bool(configs.train_val_loader_drop_last)
        batch_size = configs.batch_size
    data_set = Data(configs=configs, flag=flag)
    loader_kwargs = {}
    if configs.dataset_name == 'P12':
        drop_last = False
        loader_kwargs['pin_memory'] = bool(configs.use_gpu and torch.cuda.is_available())
        if flag == 'train':
            sampler_rng = torch.Generator().manual_seed(configs.seed)
            loader_kwargs['sampler'] = RandomSampler(data_set, generator=sampler_rng)
            loader_kwargs['generator'] = torch.Generator().manual_seed(configs.seed + 1)
        else:
            loader_kwargs['num_workers'] = 0
        shuffle_flag = False
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=loader_kwargs.pop('num_workers', configs.num_workers), drop_last=drop_last, collate_fn=collate_fn, **loader_kwargs)
    return (data_set, data_loader)
