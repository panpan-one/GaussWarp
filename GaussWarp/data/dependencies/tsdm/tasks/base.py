__all__ = ['BaseTask']
import logging
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import Any, ClassVar, Generic, Optional
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from data.dependencies.tsdm.datasets import Dataset
from data.dependencies.tsdm.encoders import ModularEncoder
from data.dependencies.tsdm.utils import LazyDict
from data.dependencies.tsdm.utils.types import KeyVar

class BaseTaskMetaClass(ABCMeta):

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f'{cls.__module__}.{cls.__name__}')
        super().__init__(*args, **kwargs)

class BaseTask(ABC, Generic[KeyVar], metaclass=BaseTaskMetaClass):
    LOGGER: ClassVar[logging.Logger]
    train_batch_size: int = 32
    eval_batch_size: int = 128
    encoder: Optional[ModularEncoder] = None
    postprocessor: Optional[ModularEncoder] = None

    def __repr__(self) -> str:
        string = f'{self.__class__.__name__}(test_metric={type(self.test_metric).__name__})'
        return string

    @property
    @abstractmethod
    def test_metric(self) -> Callable[..., Tensor]:
        pass

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DataFrame:
        pass

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyVar]:
        pass

    @property
    @abstractmethod
    def splits(self) -> Mapping[KeyVar, Any]:
        pass

    @abstractmethod
    def get_dataloader(self, key: KeyVar, /, **dataloader_kwargs: Any) -> DataLoader:
        pass

    @cached_property
    def dataloaders(self) -> Mapping[Any, DataLoader]:
        kwargs: dict[Any, Any] = {'batch_size': self.eval_batch_size, 'shuffle': False, 'drop_last': False}
        return LazyDict({key: (self.get_dataloader, kwargs | {'key': key}) for key in self.splits})
