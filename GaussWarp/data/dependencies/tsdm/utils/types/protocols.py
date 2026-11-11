from __future__ import annotations
__all__ = ['Array', 'NTuple']
from collections.abc import Iterable, Sequence
from typing import Protocol, TypeAlias, TypeVar, Union, overload, runtime_checkable
ScalarType = TypeVar('ScalarType')
T = TypeVar('T')
Either: TypeAlias = Union[T, 'Array[T]']
Self = TypeVar('Self', bound='Array')

@runtime_checkable
class Array(Protocol[ScalarType]):

    @property
    def shape(self) -> Iterable[int]:
        ...

    @property
    def dtype(self) -> type[ScalarType]:
        ...

    def __len__(self) -> int:
        ...

    @overload
    def __getitem__(self, key: int) -> ScalarType:
        ...

    @overload
    def __getitem__(self, key: Sequence[bool]) -> Array[ScalarType]:
        ...

    @overload
    def __getitem__(self, key: Sequence[int]) -> Array[ScalarType]:
        ...

    def __getitem__(self, key):
        ...

    def __eq__(self, other: object) -> Array[bool]:
        ...

    def __le__(self, other: object) -> Array[bool]:
        ...

    def __ge__(self, other: object) -> Array[bool]:
        ...

    def __lt__(self, other: object) -> Array[bool]:
        ...

    def __gt__(self, other: object) -> Array[bool]:
        ...

    def __ne__(self, other: object) -> Array[bool]:
        ...

    def __neg__(self) -> Array[ScalarType]:
        ...

    def __invert__(self) -> Array[ScalarType]:
        ...

    def __add__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __radd__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __iadd__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __sub__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __rsub__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __isub__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __mul__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __rmul__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __imul__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __truediv__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __rtruediv__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

    def __itruediv__(self, other: Either[ScalarType]) -> Array[ScalarType]:
        ...

@runtime_checkable
class NTuple(Protocol):

    @property
    def _fields(self) -> tuple[str, ...]:
        return ()

    def _asdict(self) -> dict[str, object]:
        ...
