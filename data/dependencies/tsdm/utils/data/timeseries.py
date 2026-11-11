from __future__ import annotations
__all__ = ['TimeTensor', 'TimeSeriesDataset', 'TimeSeriesTuple', 'TimeSeriesBatch', 'IndexedArray']
from collections import namedtuple
from collections.abc import Mapping, Sized
from typing import Any, NamedTuple, Optional, TypeAlias, Union
import numpy as np
from pandas import DataFrame, Index, Series, Timedelta
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from data.dependencies.tsdm.utils.strings import repr_array, repr_sequence

class _IndexMethodClone:

    def __init__(self, data: Tensor, index: Index, method: str='loc'):
        self.data = data
        self.index = index
        self.index_method = getattr(self.index, method)

    def __getitem__(self, key):
        idx = self.index_method[key]
        if isinstance(idx, Series):
            idx = idx.values
        return self.data[idx]

class _TupleIndexMethodClone:

    def __init__(self, data: tuple[Tensor, ...], index: tuple[Index, ...], method: str='loc'):
        self.data = data
        self.index = index
        self.method = tuple((getattr(idx, method) for idx in self.index))

    def __getitem__(self, item):
        indices = tuple((method[item] for method in self.method))
        return tuple((data[indices] for data in self.data))

class TimeTensor(Tensor):

    def __new__(cls, x: Sized, *args: Any, index: Optional[Index]=None, **kwargs: Any) -> TimeTensor:
        if isinstance(x, (DataFrame, Series)):
            assert index is None, 'Index given, but x is DataFrame/Series'
            x = x.values
        return super().__new__(cls, *(x, *args), **kwargs)

    def __init__(self, x: Sized, index: Optional[Index]=None):
        super().__init__()
        if isinstance(x, (DataFrame, Series)):
            index = x.index
        else:
            index = Index(np.arange(len(x))) if index is None else index
        self.index = Series(np.arange(len(x)), index=index)
        self.loc = _IndexMethodClone(self, self.index, 'loc')
        self.iloc = _IndexMethodClone(self, self.index, 'iloc')
        self.at = _IndexMethodClone(self, self.index, 'at')
        self.iat = _IndexMethodClone(self, self.index, 'iat')
IndexedArray: TypeAlias = Series | DataFrame | TimeTensor

def is_indexed_array(x: Any) -> bool:
    return isinstance(x, (DataFrame, Series, TimeTensor))

class TimeSeriesTuple(NamedTuple):
    timestamps: Tensor
    observables: Tensor
    covariates: Tensor
    targets: Tensor

class TimeSeriesBatch(NamedTuple):
    inputs: TimeSeriesTuple
    future: TimeSeriesTuple
    metadata: Tensor

class TimeSeriesDataset(TorchDataset):
    timeseries: IndexedArray | tuple[IndexedArray, ...]
    metadata: Optional[IndexedArray | tuple[IndexedArray, ...]] = None
    ts_type: type[tuple] = tuple
    md_type: type[tuple] = tuple

    def __init__(self, timeseries: Union[IndexedArray, tuple[IndexedArray, ...], Mapping[str, IndexedArray]], metadata: Optional[Any | tuple[Any, ...] | dict[str, Any]]=None):
        super().__init__()
        if isinstance(timeseries, Mapping):
            self.ts_type = namedtuple('timeseries', timeseries.keys())
            self.timeseries = self.ts_type(**timeseries)
        elif isinstance(timeseries, tuple):
            if hasattr(timeseries, '_fields'):
                self.ts_type = type(timeseries)
                self.timeseries = timeseries
            else:
                self.ts_type = tuple
                self.timeseries = tuple(timeseries)
        else:
            self.timeseries = timeseries
        if isinstance(metadata, Mapping):
            self.md_type = namedtuple('metadata', metadata.keys())
            self.timeseries = self.md_type(**timeseries)
        elif isinstance(metadata, tuple):
            if hasattr(metadata, '_fields'):
                self.md_type = type(metadata)
                self.metadata = metadata
            else:
                self.md_type = tuple
                self.metadata = tuple(metadata)
        else:
            self.metadata = metadata

    def __repr__(self) -> str:
        title = self.__class__.__name__
        pad = 2
        if isinstance(self.timeseries, tuple):
            ts_lines = repr_sequence(self.timeseries, linebreaks=False, padding=pad, repr_fun=repr_array, title='timeseries=')
        else:
            ts_lines = repr_array(self.timeseries)
        if self.metadata is None:
            md_lines = f'{None}'
        elif isinstance(self.metadata, tuple):
            md_lines = repr_sequence(self.metadata, linebreaks=False, padding=pad, repr_fun=repr_array, title='metadata=')
        else:
            md_lines = 'metadata=' + repr_array(self.metadata)
        return f'{title}[{ts_lines}, {md_lines}]'

    def __len__(self) -> int:
        if isinstance(self.timeseries, tuple):
            return sum((len(ts) for ts in self.timeseries))
        return len(self.timeseries)

    def timespan(self) -> Timedelta:
        if isinstance(self.timeseries, tuple):
            tmax = max((max(ts.index) for ts in self.timeseries))
            tmin = min((min(ts.index) for ts in self.timeseries))
            return tmax - tmin
        return max(self.timeseries.index) - min(self.timeseries.index)

    def __getitem__(self, item):
        if isinstance(self.timeseries, tuple):
            if hasattr(self.timeseries, '_fields'):
                timeseries = self.ts_type(*(ts[item] for ts in self.timeseries))
            else:
                timeseries = tuple((ts[item] for ts in self.timeseries))
        else:
            timeseries = self.timeseries.loc[item]
        return TimeSeriesDataset(timeseries, metadata=self.metadata)

    def __iter__(self):
        yield self.timeseries
        yield self.metadata
