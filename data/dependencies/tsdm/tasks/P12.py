__all__ = ['MIMIC_IV_Bilos2021', 'mimic_collate', 'Sample', 'Batch', 'TaskDataset']
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple
import torch
from pandas import DataFrame, Index, MultiIndex
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch import nan as NAN
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from data.dependencies.tsdm.datasets import Physionet2012 as Physionet2012_Dataset
from data.dependencies.tsdm.encoders import FrameEncoder, MinMaxScaler, Standardizer
from data.dependencies.tsdm.tasks.base import BaseTask
from data.dependencies.tsdm.utils import is_partition
from data.dependencies.tsdm.utils.strings import repr_namedtuple
import pandas as pd
import numpy as np

class Inputs(NamedTuple):
    t: Tensor
    x: Tensor
    t_target: Tensor

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)

class Sample(NamedTuple):
    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)

class Batch(NamedTuple):
    x_time: Tensor
    x_vals: Tensor
    x_mask: Tensor
    y_time: Tensor
    y_vals: Tensor
    y_mask: Tensor

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)

@dataclass
class TaskDataset(Dataset):
    tensors: list[tuple[Tensor, Tensor]]
    observation_time: float
    prediction_steps: int
    idx_list: list

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, key: int) -> Sample:
        t, x = self.tensors[key]
        observations = t <= self.observation_time
        first_target = observations.sum()
        sample_mask = slice(0, first_target)
        target_mask = slice(first_target, first_target + self.prediction_steps)
        return Sample(key=self.idx_list[key], inputs=Inputs(t[sample_mask], x[sample_mask], t[target_mask]), targets=x[target_mask], originals=(t, x))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

def physionet_collate(batch: list[Sample]) -> Batch:
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []
    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets
        time = torch.cat((t, t_target))
        sorted_idx = torch.argsort(time)
        x_padding = torch.full((t_target.shape[0], x.shape[-1]), fill_value=NAN, device=x.device)
        values = torch.cat((x, x_padding))
        mask_y = y.isfinite()
        mask_pad = torch.zeros_like(x, dtype=torch.bool)
        mask_x = torch.cat((mask_pad, mask_y))
        x_vals.append(values[sorted_idx])
        x_time.append(time[sorted_idx])
        x_mask.append(mask_x[sorted_idx])
        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)
    x_time.append(torch.zeros(40))
    y_time.append(torch.zeros(3))
    x_vals.append(torch.zeros(40, 36))
    y_vals.append(torch.zeros(3, 36))
    x_mask.append(torch.zeros(40, 36))
    y_mask.append(torch.zeros(3, 36))
    x_time = pad_sequence(x_time, batch_first=True)
    x_vals = pad_sequence(x_vals, batch_first=True, padding_value=float('nan'))
    x_mask = pad_sequence(x_mask, batch_first=True)
    y_time = pad_sequence(y_time, batch_first=True)
    y_vals = pad_sequence(y_vals, batch_first=True, padding_value=float('nan'))
    y_mask = pad_sequence(y_mask, batch_first=True)
    x_time = x_time[:-1]
    y_time = y_time[:-1]
    x_vals = x_vals[:-1]
    y_vals = y_vals[:-1]
    x_mask = x_mask[:-1]
    y_mask = y_mask[:-1]
    x = x_vals
    y = y_vals
    B, L, N = y.shape
    B, L_, N = x.shape
    shuffle = False
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    mask_train = torch.where(torch.isnan(x), 0, 1)
    extra_mask = torch.rand_like(mask_train.float())
    extra_mask = torch.where(extra_mask <= 0.1, 0, 1)
    mask_train *= extra_mask
    x_ = x.clone()
    x_[:, -L:, :] = y
    mask_observe = torch.where(torch.isnan(x_), 0, 1)
    mask_observe *= extra_mask
    have_nan = False
    if have_nan == True:
        res = x_[:, :, idx]
    else:
        res = torch.nan_to_num(x_)[:, :, idx]
    res *= extra_mask
    assert x_ is not None
    return {'observed_data': res, 'unshuffled': torch.nan_to_num(x_), 'observed_mask': mask_observe[:, :, idx], 'mask_train': mask_train[:, :, idx], 'time_steps': x_time, 'idx': idx, 'feature_representation': torch.tensor(idx), 'extra_mask': extra_mask, 'keys': [sample.key for sample in batch]}

class Physionet2012(BaseTask):
    RANDOM_STATE = 0
    test_size = 0.1
    valid_size = 0.1
    STATISTICS_CUTOFF_HOURS = 35.5
    TIME_SCALE_HOURS = 48.0
    encoder: FrameEncoder[Standardizer, dict[Any, MinMaxScaler]]

    def __init__(self, normalize_time: bool=True, seq_len: int=36, pred_len: int=3, num_folds: int=1):
        super().__init__()
        self.prediction_steps = pred_len
        self.observation_time = seq_len
        self.num_folds = num_folds
        self.encoder = FrameEncoder(column_encoders=Standardizer(), index_encoders={'Time': MinMaxScaler()})
        self.normalize_time = normalize_time
        _ = self.dataset

    def get_mean_value_according_ID(self, IDs):
        ds = Physionet2012_Dataset()
        ts_ = ds.dataset
        tsa = ts_['A'][0]
        tsb = ts_['B'][0]
        tsc = ts_['C'][0]
        ts = pd.concat([tsa, tsb, tsc])
        datas = []
        for id in IDs:
            data = ts.loc[id]
            data = np.nanmean(data, axis=0)
            datas.append(np.expand_dims(data, 0))
        datas = np.concatenate(datas, axis=0)
        datas = np.nanmean(datas, axis=0)
        cols = list(ts.columns)
        tsa = ts_['A'][1]
        tsb = ts_['B'][1]
        tsc = ts_['C'][1]
        ts = pd.concat([tsa, tsb, tsc])
        datas_ = []
        for id in IDs:
            data = ts.loc[id]
            datas_.append(np.expand_dims(data, 0))
        datas_ = np.concatenate(datas_, axis=0)
        datas_ = np.nanmean(datas_, axis=0)
        res = np.concatenate([datas_, datas], axis=0)
        cols_ = list(ts.columns)
        cols_.extend(cols)
        return (res, cols_)

    @cached_property
    def dataset(self) -> DataFrame:
        ds = Physionet2012_Dataset()
        ts_ = ds.dataset
        tsa = ts_['A'][0]
        tsb = ts_['B'][0]
        tsc = ts_['C'][0]
        ts = pd.concat([tsa, tsb, tsc])
        self.IDs = ts.reset_index()['RecordID'].unique()
        train_history = ts.loc[self.folds[0]['train']]
        train_history = train_history.loc[train_history.index.get_level_values('Time') <= self.STATISTICS_CUTOFF_HOURS].astype(np.float64)
        self.encoder.fit(train_history)
        value_encoder = self.encoder.column_encoders
        value_encoder.mean = np.asarray(value_encoder.mean, dtype=np.float32)
        value_encoder.stdv = np.asarray(value_encoder.stdv, dtype=np.float32)
        retained_columns = ts.columns[np.isfinite(value_encoder.mean) & np.isfinite(value_encoder.stdv) & (value_encoder.stdv > 1e-08)]
        index_encoder = self.encoder.index_encoders['Time']
        index_encoder.fit(pd.Series([0.0, self.TIME_SCALE_HOURS], name='Time'))
        ts = self.encoder.encode(ts.astype(np.float32))
        self.observation_time /= self.TIME_SCALE_HOURS
        ts = ts.loc[:, retained_columns]
        return ts.where((-5 < ts) & (ts < 5)).copy()

    @cached_property
    def folds(self) -> list[dict[str, Sequence[int]]]:
        num_folds = 5
        folds = []
        for _ in range(num_folds):
            train_idx, test_idx = train_test_split(self.IDs, test_size=self.test_size, random_state=self.RANDOM_STATE, shuffle=False)
            train_idx, valid_idx = train_test_split(train_idx, test_size=self.valid_size, random_state=self.RANDOM_STATE, shuffle=False)
            fold = {'train': train_idx, 'val': valid_idx, 'test': test_idx}
            assert is_partition(fold.values(), union=self.IDs)
            folds.append(fold)
        return folds

    @cached_property
    def split_idx(self):
        fold_idx = Index(list(range(len(self.folds))), name='fold')
        splits = DataFrame(index=self.IDs, columns=fold_idx, dtype='string')
        for k in range(self.num_folds):
            for key, split in self.folds[k].items():
                mask = splits.index.isin(split)
                splits[k] = splits[k].where(~mask, key)
        return splits

    @cached_property
    def split_idx_sparse(self) -> DataFrame:
        df = self.split_idx
        columns = df.columns
        categories = {col: df[col].astype('category').dtype.categories for col in columns}
        if isinstance(df.columns, MultiIndex):
            index_tuples = [(*col, cat) for col, cats in zip(columns, categories) for cat in categories[col]]
            names = df.columns.names + ['partition']
        else:
            index_tuples = [(col, cat) for col, cats in zip(columns, categories) for cat in categories[col]]
            names = [df.columns.name, 'partition']
        new_columns = MultiIndex.from_tuples(index_tuples, names=names)
        result = DataFrame(index=df.index, columns=new_columns, dtype=bool)
        if isinstance(df.columns, MultiIndex):
            for col in new_columns:
                result[col] = df[col[:-1]] == col[-1]
        else:
            for col in new_columns:
                result[col] = df[col[0]] == col[-1]
        return result

    @cached_property
    def test_metric(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.MSELoss()

    @cached_property
    def splits(self) -> Mapping:
        splits = {}
        for key in self.index:
            mask = self.split_idx_sparse[key]
            ids = self.split_idx_sparse.index[mask]
            splits[key] = self.dataset.loc[ids]
        return splits

    @cached_property
    def index(self) -> MultiIndex:
        return self.split_idx_sparse.columns

    @cached_property
    def tensors(self) -> Mapping:
        tensors = {}
        for _id in self.IDs:
            s = self.dataset.loc[_id]
            t = torch.tensor(s.index.values, dtype=torch.float32)
            x = torch.tensor(s.values, dtype=torch.float32)
            tensors[_id] = (t, x)
        return tensors

    def get_dataloader(self, key: tuple[int, str], /, **dataloader_kwargs: Any) -> DataLoader:
        fold, partition = key
        fold_idx = self.folds[fold][partition]
        dataset = TaskDataset([val for idx, val in self.tensors.items() if idx in fold_idx], observation_time=self.observation_time, prediction_steps=self.prediction_steps, idx_list=fold_idx)
        kwargs: dict[str, Any] = {'collate_fn': lambda *x: x} | dataloader_kwargs
        kwargs: dict[str, Any] = {'collate_fn': physionet_collate} | dataloader_kwargs
        return DataLoader(dataset, **kwargs)

    def get_dataset(self, key: tuple[int, str]) -> Dataset:
        fold, partition = key
        fold_idx = self.folds[fold][partition]
        dataset = TaskDataset([val for idx, val in self.tensors.items() if idx in fold_idx], observation_time=self.observation_time, prediction_steps=self.prediction_steps, idx_list=fold_idx)
        return dataset
