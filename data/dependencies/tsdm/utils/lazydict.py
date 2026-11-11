__all__ = ['LazyDict', 'LazyFunction']
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from typing import Any, Generic, NamedTuple, Union, overload
from data.dependencies.tsdm.utils.types import KeyVar, ObjectVar
__logger__ = logging.getLogger(__name__)

class LazyFunction(NamedTuple):
    func: Callable[..., Any]
    args: Iterable[Any] = ()
    kwargs: Mapping[str, Any] = {}

    def __call__(self) -> Any:
        return self.func(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return f'<LazyFunction: {self.func.__name__}>'

class LazyDict(MutableMapping[KeyVar, ObjectVar], Generic[KeyVar, ObjectVar]):

    @staticmethod
    def _validate_value(value: Any) -> LazyFunction:
        if isinstance(value, LazyFunction):
            return value
        if callable(value):
            return LazyFunction(func=value)
        if isinstance(value, tuple):
            if len(value) < 1 or not callable(value[0]):
                raise ValueError('Invalid tuple')
            func = value[0]
            if len(value) == 1:
                return LazyFunction(func)
            if len(value) == 2 and isinstance(value[1], Mapping):
                return LazyFunction(func, kwargs=value[1])
            if len(value) == 2 and isinstance(value[1], Iterable):
                return LazyFunction(func, args=value[1])
            if len(value) == 3 and isinstance(value[1], Iterable) and isinstance(value[2], Mapping):
                return LazyFunction(func, args=value[1], kwargs=value[2])
            raise ValueError('Invalid tuple')
        raise ValueError('Invalid value')

    @overload
    def __init__(self, /, **kwargs: ObjectVar) -> None:
        ...

    @overload
    def __init__(self, mapping: Mapping[KeyVar, Union[tuple[Callable[..., ObjectVar], tuple], tuple[Callable[..., ObjectVar], dict], tuple[Callable[..., ObjectVar], tuple, dict]]], /, **kwargs: ObjectVar) -> None:
        ...

    @overload
    def __init__(self, iterable: Iterable[tuple[KeyVar, Union[tuple[Callable[..., ObjectVar], tuple], tuple[Callable[..., ObjectVar], dict], tuple[Callable[..., ObjectVar], tuple, dict]]]], /, **kwargs: ObjectVar) -> None:
        ...

    def __init__(self, /, *args: Any, **kwargs: ObjectVar) -> None:
        self._dict: dict[Any, Any] = {}
        self._initialized: dict[Any, bool] = {}
        if len(args) > 1:
            raise TypeError('Too many positional arguments')
        if len(args) == 0:
            self.update(**kwargs)
            return
        arg = args[0]
        if isinstance(arg, Mapping):
            self.update(**arg)
        elif isinstance(arg, Iterable):
            for item in arg:
                if not isinstance(item, tuple) and len(item) == 2:
                    raise ValueError('Invalid iterable')
                key, value = item
                self[key] = value

    def _initialize(self, key: KeyVar) -> None:
        __logger__.info('%s: Initializing %s', self, key)
        if key not in self._dict:
            raise KeyError(key)
        if not self._initialized[key]:
            self._dict[key] = self._dict[key]()
            self._initialized[key] = True

    def __getitem__(self, key: KeyVar) -> ObjectVar:
        if key not in self._dict:
            raise KeyError(key)
        if not self._initialized[key]:
            value = self._dict[key]
            func, args, kwargs = value
            self._dict[key] = func(*args, **kwargs)
            self._initialized[key] = True
        return self._dict[key]

    def __setitem__(self, key: KeyVar, value: ObjectVar) -> None:
        self._dict[key] = self._validate_value(value)
        self._initialized[key] = False

    def __delitem__(self, key: KeyVar) -> None:
        del self._dict[key]
        del self._initialized[key]

    def __iter__(self) -> Iterator[KeyVar]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        padding = ' ' * 2
        max_key_length = max((len(str(key)) for key in self.keys()))
        items = [(str(key), self._dict.get(key)) for key in self]
        maxitems = 10
        string = self.__class__.__name__ + '(\n'
        if maxitems is None or len(self) <= maxitems:
            string += ''.join((f'{padding}{str(key):<{max_key_length}}: {value}\n' for key, value in items))
        else:
            string += ''.join((f'{padding}{str(key):<{max_key_length}}: {value}\n' for key, value in items[:maxitems // 2]))
            string += f'{padding}...\n'
            string += ''.join((f'{padding}{str(key):<{max_key_length}}: {value}\n' for key, value in items[-maxitems // 2:]))
        string += ')'
        return string

    def __str__(self):
        return str(self._dict)
