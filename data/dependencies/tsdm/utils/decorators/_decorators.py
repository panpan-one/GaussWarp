__all__ = ['decorator', 'timefun', 'trace', 'vectorize', 'wrap_func', 'autojit', 'IterItems', 'IterKeys', 'DecoratorError']
import gc
import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, Signature, signature
from time import perf_counter_ns
from typing import Any, Optional, overload
from torch import jit, nn
from data.dependencies.tsdm.config import conf
from data.dependencies.tsdm.utils.types import AnyTypeVar, ClassVar, ObjectVar, ReturnVar, TorchModuleVar
from data.dependencies.tsdm.utils.types.abc import CollectionType
__logger__ = logging.getLogger(__name__)
KEYWORD_ONLY = Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = Parameter.VAR_KEYWORD
VAR_POSITIONAL = Parameter.VAR_POSITIONAL
EMPTY = Parameter.empty
_DECORATED = object()
PARAM_TYPES = ((POSITIONAL_ONLY, True), (POSITIONAL_ONLY, False), (POSITIONAL_OR_KEYWORD, True), (POSITIONAL_OR_KEYWORD, False), (VAR_POSITIONAL, True), (KEYWORD_ONLY, True), (KEYWORD_ONLY, False), (VAR_KEYWORD, True))

def rpartial(func: Callable, /, *fixed_args: Any, **fixed_kwargs: Any) -> Callable:

    @wraps(func)
    def _wrapper(*func_args, **func_kwargs):
        return func(*func_args + fixed_args, **func_kwargs | fixed_kwargs)
    return _wrapper

@dataclass
class DecoratorError(Exception):
    decorated: Callable
    message: str = ''

    def __call__(self, *message_lines):
        return DecoratorError(self.decorated, message='\n'.join(message_lines))

    def __str__(self):
        sign = signature(self.decorated)
        max_key_len = max(9, max((len(key) for key in sign.parameters)))
        max_kind_len = max((len(str(param.kind)) for param in sign.parameters.values()))
        default_message: tuple[str, ...] = (f'Signature: {sign}', '\n'.join((f'{key.ljust(max_key_len)}: {str(param.kind).ljust(max_kind_len)}, Optional: {param.default is EMPTY}' for key, param in sign.parameters.items())), self.message)
        return super().__str__() + '\n' + '\n'.join(default_message)

def _last_positional_only_arg_index(sig: Signature) -> int:
    for i, param in enumerate(sig.parameters.values()):
        if param.kind is not POSITIONAL_ONLY:
            return i
    return len(sig.parameters)

def decorator(deco: Callable) -> Callable:
    __logger__.debug('>>>>> Creating decorator %s <<<<<', deco)
    deco_sig = signature(deco)
    ErrorHandler = DecoratorError(deco)
    BUCKETS: dict[Any, set[str]] = {key: set() for key in PARAM_TYPES}
    for key, param in deco_sig.parameters.items():
        BUCKETS[param.kind, param.default is EMPTY].add(key)
    __logger__.debug('DETECTED SIGNATURE:\n\t%s POSITIONAL_ONLY       (mandatory)\n\t%s POSITIONAL_ONLY       (optional)\n\t%s POSITIONAL_OR_KEYWORD (mandatory)\n\t%s POSITIONAL_OR_KEYWORD (optional)\n\t%s VAR_POSITIONAL\n\t%s KEYWORD_ONLY          (mandatory)\n\t%s KEYWORD_ONLY          (optional)\n\t%s VAR_KEYWORD', *(len(BUCKETS[key]) for key in PARAM_TYPES))
    if BUCKETS[POSITIONAL_OR_KEYWORD, True] or BUCKETS[POSITIONAL_OR_KEYWORD, False]:
        raise ErrorHandler('Decorator does not support POSITIONAL_OR_KEYWORD arguments!!', "Separate positional and keyword arguments using '/' and '*':>>> def deco(func, /, *, ko1, ko2, **kwargs): ...", 'Cf. https://www.python.org/dev/peps/pep-0570/')
    if BUCKETS[POSITIONAL_ONLY, False]:
        raise ErrorHandler('Decorator does not support POSITIONAL_ONLY arguments with defaults!!')
    if not len(BUCKETS[POSITIONAL_ONLY, True]) == 1:
        raise ErrorHandler('Decorator must have exactly 1 POSITIONAL_ONLY argument: the function to be decorated.>>> def deco(func, /, *, ko1, ko2, **kwargs): ...')
    if BUCKETS[VAR_POSITIONAL, True]:
        raise ErrorHandler('Decorator does not support VAR_POSITIONAL arguments!!')

    @wraps(deco)
    def _parametrized_decorator(__func__: Optional[Callable]=None, *args: Any, **kwargs: Any) -> Callable:
        __logger__.debug('DECORATING \n\tfunc=%s: \n\targs=%s, \n\tkwargs=%s', deco, args, kwargs)
        if __func__ is None:
            __logger__.debug('%s: Decorator used in BRACKET mode.', deco)
            return rpartial(deco, *args, **kwargs)
        assert callable(__func__), 'First argument must be callable!'
        __logger__.debug('%s: Decorator in FUNCTIONAL/BARE mode.', deco)
        return deco(*(__func__, *args), **kwargs)
    return _parametrized_decorator

def attribute(func):

    @wraps(func, updated=())
    class _attribute:
        __slots__ = ('func', 'payload')
        sentinel = object()

        def __init__(self, function):
            self.func = function
            self.payload = self.sentinel

        def __get__(self, obj, obj_type=None):
            if obj is None:
                return self
            if self.payload is self.sentinel:
                self.payload = self.func(obj)
            return self.payload
    return _attribute(func)

@decorator
def timefun(fun: Callable[..., ReturnVar], /, *, append: bool=False, loglevel: int=logging.WARNING) -> Callable[..., ReturnVar]:
    timefun_logger = logging.getLogger('timefun')

    @wraps(fun)
    def _timed_fun(*args, **kwargs):
        gc.collect()
        gc.disable()
        try:
            start_time = perf_counter_ns()
            result = fun(*args, **kwargs)
            end_time = perf_counter_ns()
            elapsed = (end_time - start_time) / 10 ** 9
            timefun_logger.log(loglevel, '%s executed in %.4f s', fun.__qualname__, elapsed)
        except (KeyboardInterrupt, SystemExit) as E:
            raise E
        except Exception as E:
            result = None
            elapsed = float('nan')
            RuntimeWarning(f'Function execution failed with Exception {E}')
            timefun_logger.log(loglevel, '%s failed with Exception %s', fun.__qualname__, E)
        gc.enable()
        return (result, elapsed) if append else result
    return _timed_fun

def trace(func: Callable[..., ReturnVar]) -> Callable[..., ReturnVar]:
    logger = logging.getLogger('trace')

    @wraps(func)
    def _wrapper(*args, **kwargs):
        logger.info('%s', '\n\t'.join((f'{func.__qualname__}: ENTERING', f'args={tuple((type(arg).__name__ for arg in args))}', f'kwargs={str({k: type(v).__name__ for k, v in kwargs.items()})}')))
        try:
            logger.info('%s: EXECUTING', func.__qualname__)
            result = func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit) as E:
            raise E
        except Exception as E:
            logger.error('%s: FAILURE with Exception %s', func.__qualname__, E)
            raise RuntimeError(f'Function execution failed with Exception {E}') from E
        else:
            logger.info('%s: SUCCESS with result=%s', func.__qualname__, type(result).__name__)
        logger.info('%s', '\n\t'.join((f'{func.__qualname__}: EXITING',)))
        return result
    return _wrapper

def autojit(base_class: type[TorchModuleVar]) -> type[TorchModuleVar]:
    assert issubclass(base_class, nn.Module)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):

        def __new__(cls, *args: Any, **kwargs: Any) -> TorchModuleVar:
            instance: TorchModuleVar = base_class(*args, **kwargs)
            if conf.autojit:
                scripted: TorchModuleVar = jit.script(instance)
                return scripted
            return instance
    assert issubclass(WrappedClass, base_class)
    return WrappedClass

@decorator
def vectorize(func: Callable[[ObjectVar], ReturnVar], /, *, kind: type[CollectionType]) -> Callable[[ObjectVar | CollectionType], ReturnVar | CollectionType]:
    params = list(signature(func).parameters.values())
    if not params:
        raise ValueError(f'{func} has no parameters')
    if params[0].kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
        raise ValueError(f'{func} must have a single positional parameter!')
    for param in params[1:]:
        if param.kind not in (Parameter.KEYWORD_ONLY, Parameter.VAR_KEYWORD):
            raise ValueError(f'{func} must have a single positional parameter!')

    @wraps(func)
    def _wrapper(arg, *args):
        if not args:
            return func(arg)
        return kind((func(x) for x in (arg, *args)))
    return _wrapper

@overload
def IterItems(obj: ClassVar) -> ClassVar:
    ...

@overload
def IterItems(obj: ObjectVar) -> ObjectVar:
    ...

def IterItems(obj: AnyTypeVar) -> AnyTypeVar:
    base_class = obj if isinstance(obj, type) else type(obj)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):

        def __getitem__(self, key: Any) -> tuple[Any, Any]:
            return (key, super().__getitem__(key))

        def __repr__(self) -> str:
            return 'IterItems@' + super().__repr__()
    if isinstance(obj, type):
        return WrappedClass
    obj = deepcopy(obj)
    obj.__class__ = WrappedClass
    return obj

@overload
def IterKeys(obj: ClassVar) -> ClassVar:
    ...

@overload
def IterKeys(obj: ObjectVar) -> ObjectVar:
    ...

def IterKeys(obj: AnyTypeVar) -> AnyTypeVar:
    base_class = obj if isinstance(obj, type) else type(obj)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):

        def __getitem__(self, key: Any) -> tuple[Any, Any]:
            return key

        def __repr__(self) -> str:
            return 'IterKeys@' + super().__repr__()
    if isinstance(obj, type):
        return WrappedClass
    obj = deepcopy(obj)
    obj.__class__ = WrappedClass
    return obj

@decorator
def wrap_func(func: Callable[..., ReturnVar], /, *, before: Optional[Callable[..., Any]]=None, after: Optional[Callable[..., Any]]=None) -> Callable[..., ReturnVar]:
    if before is None and after is None:
        __logger__.debug('No hooks added to %s', func)
        return func
    if before is not None and after is None:
        __logger__.debug('Adding pre hook %s to %s', before, func)

        @wraps(func)
        def _wrapper(*args, **kwargs):
            before(*args, **kwargs)
            result = func(*args, **kwargs)
            return result
        return _wrapper
    if before is None and after is not None:
        __logger__.debug('Adding post hook %s to %s', after, func)

        @wraps(func)
        def _wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            after(*args, **kwargs)
            return result
        return _wrapper
    if before is not None and after is not None:
        __logger__.debug('Adding pre hook %s to %s', before, func)
        __logger__.debug('Adding post hook %s to %s', after, func)

        @wraps(func)
        def _wrapper(*args, **kwargs):
            before(*args, **kwargs)
            result = func(*args, **kwargs)
            after(*args, **kwargs)
            return result
        return _wrapper
    raise RuntimeError(f'Unreachable code reached for {func}')
