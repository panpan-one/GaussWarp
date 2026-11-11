__all__ = ['AnyTypeVar', 'AliasVar', 'ClassVar', 'DtypeVar', 'KeyVar', 'ModuleVar', 'ObjectVar', 'PandasVar', 'PathVar', 'ReturnVar', 'Self', 'TensorVar', 'TorchModuleVar', 'ValueVar', 'Args', 'KWArgs', 'Nested', 'PandasObject', 'PathType', 'Parameters']
import os
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from types import GenericAlias, ModuleType
from typing import Any, ParamSpec, TypeAlias, TypeVar
from numpy import ndarray
from pandas import DataFrame, Index, Series
from torch import Tensor, nn
Parameters = ParamSpec('Parameters')
AnyTypeVar = TypeVar('AnyTypeVar')
AliasVar = TypeVar('AliasVar', bound=GenericAlias)
ClassVar = TypeVar('ClassVar', bound=type)
DtypeVar = TypeVar('DtypeVar')
KeyVar = TypeVar('KeyVar', bound=Hashable)
ModuleVar = TypeVar('ModuleVar', bound=ModuleType)
ObjectVar = TypeVar('ObjectVar', bound=object)
PandasVar = TypeVar('PandasVar', Index, Series, DataFrame)
PathVar = TypeVar('PathVar', str, Path, os.PathLike[str])
ReturnVar = TypeVar('ReturnVar')
Self = TypeVar('Self')
TensorVar = TypeVar('TensorVar', Tensor, ndarray)
TorchModuleVar = TypeVar('TorchModuleVar', bound=nn.Module)
ValueVar = TypeVar('ValueVar')
Args: TypeAlias = tuple[AnyTypeVar, ...]
KWArgs: TypeAlias = dict[str, AnyTypeVar]
Nested: TypeAlias = AnyTypeVar | Collection[AnyTypeVar] | Mapping[Any, AnyTypeVar]
PandasObject: TypeAlias = DataFrame | Series | Index
PathType: TypeAlias = str | Path
