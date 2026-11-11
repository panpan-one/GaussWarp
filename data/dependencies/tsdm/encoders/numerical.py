from __future__ import annotations
__all__ = ['Standardizer', 'MinMaxScaler', 'LogEncoder', 'TensorSplitter', 'TensorConcatenator']
from functools import singledispatchmethod
from typing import Any, Generic, NamedTuple, Optional, TypeAlias, TypeVar, overload
import numpy as np
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from torch import Tensor
from data.dependencies.tsdm.encoders.base import BaseEncoder
from data.dependencies.tsdm.utils.strings import repr_namedtuple
from data.dependencies.tsdm.utils.types import PandasObject
TensorLike: TypeAlias = Tensor | NDArray | DataFrame | Series
TensorType = TypeVar('TensorType', Tensor, np.ndarray, DataFrame, Series)

def get_broadcast(data: Any, /, *, axis: tuple[int, ...] | None) -> tuple[slice | None, ...]:
    rank = len(data.shape)
    if axis is None:
        return (slice(None),) * rank
    axis = tuple((a % rank for a in axis))
    broadcast = tuple((slice(None) if a in axis else None for a in range(rank)))
    return broadcast

class Standardizer(BaseEncoder, Generic[TensorType]):
    mean: Any
    stdv: Any
    ignore_nan: bool = True
    axis: tuple[int, ...] | None

    class Parameters(NamedTuple):
        mean: TensorLike
        stdv: TensorLike
        axis: tuple[int, ...] | None

        def __repr__(self) -> str:
            return repr_namedtuple(self)

    def __init__(self, /, mean: Optional[Tensor]=None, stdv: Optional[Tensor]=None, *, ignore_nan: bool=True, axis: Optional[int | tuple[int, ...]]=None):
        super().__init__()
        self.ignore_nan = ignore_nan
        self.axis = (axis,) if isinstance(axis, int) else axis
        self.mean = mean
        self.stdv = stdv

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis})'

    def __getitem__(self, item: Any) -> Standardizer:
        axis = self.axis if self.axis is None else self.axis[1:]
        encoder = Standardizer(mean=self.mean[item], stdv=self.stdv[item], axis=axis)
        encoder._is_fitted = self._is_fitted
        return encoder

    @property
    def param(self) -> Parameters:
        return self.Parameters(self.mean, self.stdv, self.axis)

    def fit(self, data: TensorType, /) -> None:
        rank = len(data.shape)
        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)
        selection = [a % rank for a in self.axis]
        axes = tuple((k for k in range(rank) if k not in selection))
        if isinstance(data, Tensor):
            if self.ignore_nan:
                self.mean = torch.nanmean(data, dim=axes)
                self.stdv = torch.sqrt(torch.nanmean((data - self.mean) ** 2, dim=axes))
            else:
                self.mean = torch.mean(data, dim=axes)
                self.stdv = torch.std(data, dim=axes)
        elif self.ignore_nan:
            self.mean = np.nanmean(data, axis=axes)
            self.stdv = np.nanstd(data, axis=axes)
        else:
            self.mean = np.mean(data, axis=axes)
            self.stdv = np.std(data, axis=axes)

    def encode(self, data: TensorType, /) -> TensorType:
        if self.mean is None:
            raise RuntimeError('Needs to be fitted first!')
        self.LOGGER.debug('Encoding data %s', data)
        broadcast = get_broadcast(data, axis=self.axis)
        self.LOGGER.debug('Broadcasting to %s', broadcast)
        data = (data - self.mean[broadcast]) / self.stdv[broadcast]
        return data

    def decode(self, data: TensorType, /) -> TensorType:
        if self.mean is None:
            raise RuntimeError('Needs to be fitted first!')
        self.LOGGER.debug('Encoding data %s', data)
        broadcast = get_broadcast(data, axis=self.axis)
        self.LOGGER.debug('Broadcasting to %s', broadcast)
        return data * self.stdv[broadcast] + self.mean[broadcast]

class MinMaxScaler(BaseEncoder, Generic[TensorType]):
    xmin: NDArray[np.number] | Tensor
    xmax: NDArray[np.number] | Tensor
    ymin: NDArray[np.number] | Tensor
    ymax: NDArray[np.number] | Tensor
    scale: NDArray[np.number] | Tensor
    axis: tuple[int, ...]

    class Parameters(NamedTuple):
        xmin: TensorLike
        xmax: TensorLike
        ymin: TensorLike
        ymax: TensorLike
        scale: TensorLike
        axis: tuple[int, ...]

        def __repr__(self) -> str:
            return repr_namedtuple(self)

    def __init__(self, /, ymin: Optional[float | TensorType]=None, ymax: Optional[float | TensorType]=None, xmin: Optional[float | TensorType]=None, xmax: Optional[float | TensorType]=None, *, axis: Optional[int | tuple[int, ...]]=None):
        super().__init__()
        self.xmin = np.array(0.0 if xmin is None else xmin)
        self.xmax = np.array(1.0 if xmax is None else xmax)
        self.ymin = np.array(0.0 if ymin is None else ymin)
        self.ymax = np.array(1.0 if ymax is None else ymax)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        self.axis = (axis,) if isinstance(axis, int) else axis

    def __getitem__(self, item: Any) -> MinMaxScaler:
        xmin = self.xmin if self.xmin.ndim == 0 else self.xmin[item]
        xmax = self.xmax if self.xmax.ndim == 0 else self.xmax[item]
        ymin = self.ymin if self.ymin.ndim == 0 else self.ymin[item]
        ymax = self.ymax if self.ymax.ndim == 0 else self.ymax[item]
        oldvals = (self.xmin, self.xmax, self.ymin, self.ymax)
        newvals = (xmin, xmax, ymin, ymax)
        assert not all((x.ndim == 0 for x in oldvals))
        lost_ranks = max((x.ndim for x in oldvals)) - max((x.ndim for x in newvals))
        encoder = MinMaxScaler(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, axis=self.axis[lost_ranks:])
        encoder._is_fitted = self._is_fitted
        return encoder

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis})'

    @property
    def param(self) -> Parameters:
        return self.Parameters(self.xmin, self.xmax, self.ymin, self.ymax, self.scale, self.axis)

    @singledispatchmethod
    def fit(self, data: TensorType, /) -> None:
        array = np.asarray(data)
        rank = len(array.shape)
        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)
        selection = [a % rank for a in self.axis]
        axes = tuple((k for k in range(rank) if k not in selection))
        self.ymin = np.array(self.ymin, dtype=array.dtype)
        self.ymax = np.array(self.ymax, dtype=array.dtype)
        self.xmin = np.nanmin(array, axis=axes)
        self.xmax = np.nanmax(array, axis=axes)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    @fit.register(torch.Tensor)
    def _(self, data: Tensor, /) -> None:
        mask = torch.isnan(data)
        self.ymin = torch.tensor(self.ymin, device=data.device, dtype=data.dtype)
        self.ymax = torch.tensor(self.ymax, device=data.device, dtype=data.dtype)
        neginf = torch.tensor(float('-inf'), device=data.device, dtype=data.dtype)
        posinf = torch.tensor(float('+inf'), device=data.device, dtype=data.dtype)
        self.xmin = torch.amin(torch.where(mask, posinf, data), dim=self.axis)
        self.xmax = torch.amax(torch.where(mask, neginf, data), dim=self.axis)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def encode(self, data: TensorType, /) -> TensorType:
        self.LOGGER.debug('Encoding data %s', data)
        broadcast = get_broadcast(data, axis=self.axis)
        self.LOGGER.debug('Broadcasting to %s', broadcast)
        xmin = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin
        return (data - xmin) * scale + ymin

    def decode(self, data: TensorType, /) -> TensorType:
        self.LOGGER.debug('Decoding data %s', data)
        broadcast = get_broadcast(data, axis=self.axis)
        self.LOGGER.debug('Broadcasting to %s', broadcast)
        xmin = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin
        return (data - ymin) / scale + xmin

class LogEncoder(BaseEncoder):
    threshold: NDArray
    replacement: NDArray

    def fit(self, data: NDArray, /) -> None:
        assert np.all(data >= 0)
        mask = data == 0
        self.threshold = data[~mask].min()
        self.replacement = np.log2(self.threshold / 2)

    def encode(self, data: NDArray, /) -> NDArray:
        result = data.copy()
        mask = data <= 0
        result[:] = np.where(mask, self.replacement, np.log2(data))
        return result

    def decode(self, data: NDArray, /) -> NDArray:
        result = 2 ** data
        mask = result < self.threshold
        result[:] = np.where(mask, 0, result)
        return result

class FloatEncoder(BaseEncoder):
    dtypes: Series = None

    def __init__(self, dtype: str='float32'):
        self.target_dtype = dtype
        super().__init__()

    def fit(self, data: PandasObject, /) -> None:
        if isinstance(data, DataFrame):
            self.dtypes = data.dtypes
        elif isinstance(data, (Series, Index)):
            self.dtypes = data.dtype
        else:
            raise TypeError(f'Cannot get dtype of {type(data)}')

    def encode(self, data: PandasObject, /) -> PandasObject:
        return data.astype(self.target_dtype)

    def decode(self, data: PandasObject, /) -> PandasObject:
        return data.astype(self.dtypes)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class IntEncoder(BaseEncoder):
    dtypes: Series = None

    def fit(self, data: PandasObject, /) -> None:
        self.dtypes = data.dtypes

    def encode(self, data: PandasObject, /) -> PandasObject:
        return data.astype('int32')

    def decode(self, data, /):
        return data.astype(self.dtypes)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class TensorSplitter(BaseEncoder):
    lengths: list[int]
    numdims: list[int]
    axis: int
    maxdim: int
    indices_or_sections: int | list[int]

    def __init__(self, *, indices_or_sections: int | list[int]=1, axis: int=0) -> None:
        super().__init__()
        self.axis = axis
        self.indices_or_sections = indices_or_sections

    @overload
    def encode(self, data: Tensor, /) -> list[Tensor]:
        ...

    @overload
    def encode(self, data: NDArray, /) -> list[NDArray]:
        ...

    def encode(self, data, /):
        if isinstance(data, Tensor):
            return torch.tensor_split(data, self.indices_or_sections, dim=self.axis)
        return np.array_split(data, self.indices_or_sections, dim=self.axis)

    @overload
    def decode(self, data: list[Tensor], /) -> Tensor:
        ...

    @overload
    def decode(self, data: list[NDArray], /) -> NDArray:
        ...

    def decode(self, data, /):
        if isinstance(data[0], Tensor):
            return torch.cat(data, dim=self.axis)
        return np.concatenate(data, axis=self.axis)

class TensorConcatenator(BaseEncoder):
    lengths: list[int]
    numdims: list[int]
    axis: int
    maxdim: int

    def __init__(self, axis: int=0) -> None:
        super().__init__()
        self.axis = axis

    def fit(self, data: tuple[Tensor, ...], /) -> None:
        self.numdims = [d.ndim for d in data]
        self.maxdim = max(self.numdims)
        arrays = [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data]
        self.lengths = [x.shape[self.axis] for x in arrays]

    def encode(self, data: tuple[Tensor, ...], /) -> Tensor:
        return torch.cat([d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data], dim=self.axis)

    def decode(self, data: Tensor, /) -> tuple[Tensor, ...]:
        result = torch.split(data, self.lengths, dim=self.axis)
        return tuple((x.squeeze() for x in result))
