from __future__ import annotations
__all__ = ['BaseEncoder', 'IdentityEncoder', 'ChainedEncoder', 'ProductEncoder', 'DuplicateEncoder', 'CloneEncoder']
import logging
from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, ClassVar, Sequence, overload
from data.dependencies.tsdm.utils.decorators import wrap_func
from data.dependencies.tsdm.utils.strings import repr_sequence
from data.dependencies.tsdm.utils.types import ObjectVar

class BaseEncoderMetaClass(ABCMeta):

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f'{cls.__module__}.{cls.__name__}')
        super().__init__(*args, **kwargs)

class BaseEncoder(ABC, metaclass=BaseEncoderMetaClass):
    LOGGER: ClassVar[logging.Logger]
    _is_fitted: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.transform = self.encode
        self.inverse_transform = self.decode

    def __init_subclass__(cls, /, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)
        cls.fit = wrap_func(cls.fit, after=cls._post_fit_hook)
        cls.encode = wrap_func(cls.encode, before=cls._pre_encode_hook)
        cls.decode = wrap_func(cls.decode, before=cls._pre_decode_hook)

    def __matmul__(self, other: BaseEncoder) -> ChainedEncoder:
        return ChainedEncoder(self, other)

    def __or__(self, other: BaseEncoder) -> ProductEncoder:
        return ProductEncoder(self, other)

    def __pow__(self, power: int) -> DuplicateEncoder:
        return DuplicateEncoder(self, power)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        self._is_fitted = value

    @property
    def is_surjective(self) -> bool:
        return True

    @property
    def is_injective(self) -> bool:
        return True

    @property
    def is_bijective(self) -> bool:
        return self.is_surjective and self.is_injective

    def fit(self, data: Any, /) -> None:
        pass

    @abstractmethod
    def encode(self, data, /):
        pass

    @abstractmethod
    def decode(self, data, /):
        pass

    def _post_fit_hook(self, *args: Any, **kwargs: Any) -> None:
        self.is_fitted = True

    def _pre_encode_hook(self, *args: Any, **kwargs: Any) -> None:
        if not self.is_fitted:
            raise RuntimeError('Encoder has not been fitted.')

    def _pre_decode_hook(self, *args: Any, **kwargs: Any) -> None:
        if not self.is_fitted:
            raise RuntimeError('Encoder has not been fitted.')

class IdentityEncoder(BaseEncoder):

    def encode(self, data: ObjectVar, /) -> ObjectVar:
        return data

    def decode(self, data: ObjectVar, /) -> ObjectVar:
        return data

class ProductEncoder(BaseEncoder, Sequence[BaseEncoder]):
    encoders: list[BaseEncoder]

    @property
    def is_fitted(self) -> bool:
        return all((e.is_fitted for e in self.encoders))

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        for encoder in self.encoders:
            encoder.is_fitted = value

    @property
    def is_surjective(self) -> bool:
        return all((e.is_surjective for e in self.encoders))

    @property
    def is_injective(self) -> bool:
        return all((e.is_injective for e in self.encoders))

    def __init__(self, *encoders: BaseEncoder, simplify: bool=True) -> None:
        super().__init__()
        self.encoders = []
        for encoder in encoders:
            if simplify and isinstance(encoder, ProductEncoder):
                for enc in encoder:
                    self.encoders.append(enc)
            else:
                self.encoders.append(encoder)

    def __len__(self) -> int:
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> BaseEncoder:
        ...

    @overload
    def __getitem__(self, index: slice) -> ProductEncoder:
        ...

    def __getitem__(self, index: int | slice) -> BaseEncoder | ProductEncoder:
        if isinstance(index, int):
            return self.encoders[index]
        if isinstance(index, slice):
            return ProductEncoder(*self.encoders[index])
        raise ValueError(f'Index {index} not supported.')

    def __repr__(self) -> str:
        return repr_sequence(self)

    def fit(self, data: tuple[Any, ...], /) -> None:
        for encoder, x in zip(self.encoders, data):
            encoder.fit(x)

    def encode(self, data: tuple[Any, ...], /) -> Sequence[Any]:
        rtype = type(data)
        return rtype((encoder.encode(x) for encoder, x in zip(self.encoders, data)))

    def decode(self, data: tuple[Any, ...], /) -> Sequence[Any]:
        rtype = type(data)
        return rtype((encoder.decode(x) for encoder, x in zip(self.encoders, data)))

class ChainedEncoder(BaseEncoder, Sequence[BaseEncoder]):
    encoders: list[BaseEncoder]

    @property
    def is_fitted(self) -> bool:
        return all((e.is_fitted for e in self.encoders))

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        for encoder in self.encoders:
            encoder.is_fitted = value

    @property
    def is_surjective(self) -> bool:
        return all((e.is_surjective for e in self.encoders))

    @property
    def is_injective(self) -> bool:
        return all((e.is_injective for e in self.encoders))

    def __init__(self, *encoders: BaseEncoder, simplify: bool=True) -> None:
        super().__init__()
        self.encoders = []
        for encoder in encoders:
            if simplify and isinstance(encoder, ChainedEncoder):
                for enc in encoder:
                    self.encoders.append(enc)
            else:
                self.encoders.append(encoder)

    def __len__(self) -> int:
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> BaseEncoder:
        ...

    @overload
    def __getitem__(self, index: slice) -> ChainedEncoder:
        ...

    def __getitem__(self, index: int | slice) -> BaseEncoder | ChainedEncoder:
        if isinstance(index, int):
            return self.encoders[index]
        if isinstance(index, slice):
            return ChainedEncoder(*self.encoders[index])
        raise ValueError(f'Index {index} not supported.')

    def __repr__(self) -> str:
        return repr_sequence(self)

    def fit(self, data: Any, /) -> None:
        for encoder in reversed(self.encoders):
            encoder.fit(data)
            data = encoder.encode(data)

    def encode(self, data, /):
        for encoder in reversed(self.encoders):
            data = encoder.encode(data)
        return data

    def decode(self, data, /):
        for encoder in self.encoders:
            data = encoder.decode(data)
        return data

class DuplicateEncoder(BaseEncoder):

    def __init__(self, encoder: BaseEncoder, n: int=1) -> None:
        super().__init__()
        self.base_encoder = encoder
        self.n = n
        self.encoder = ProductEncoder(*(self.base_encoder for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def __repr__(self) -> str:
        return f'Duplicates[{self.n}]@{repr(self.base_encoder)}'

    def fit(self, data: Any, /) -> None:
        return self.encoder.fit(data)

    def encode(self, data, /):
        return self.encoder.encode(data)

    def decode(self, data, /):
        return self.encoder.decode(data)

class CloneEncoder(BaseEncoder):

    def __init__(self, encoder: BaseEncoder, n: int=1) -> None:
        super().__init__()
        self.base_encoder = encoder
        self.n = n
        self.encoder = ProductEncoder(*(deepcopy(self.base_encoder) for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def __repr__(self) -> str:
        return f'Copies[{self.n}]@{repr(self.base_encoder)}'

    def fit(self, data: Any, /) -> None:
        return self.encoder.fit(data)

    def encode(self, data, /):
        return self.encoder.encode(data)

    def decode(self, data, /):
        return self.encoder.decode(data)
