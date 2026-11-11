__all__ = ['DTVar', 'TDVar', 'TimeVar', 'RealDTVar', 'RealTDVar', 'RealTimeVar', 'DT', 'TD', 'Time', 'RealDT', 'RealTD', 'RealTime', 'NumpyTDVar', 'NumpyTimeVar', 'NumpyDTVar']
from datetime import datetime as py_dt
from datetime import timedelta as py_td
from typing import TypeAlias, TypeVar, Union
from numpy import datetime64 as np_dt
from numpy import floating as np_float
from numpy import integer as np_int
from numpy import timedelta64 as np_td
from pandas import Timedelta as pd_td
from pandas import Timestamp as pd_dt
DTVar = TypeVar('DTVar', int, float, np_int, np_float, np_dt, pd_dt)
TDVar = TypeVar('TDVar', int, float, np_int, np_float, np_td, pd_td)
TimeVar = TypeVar('TimeVar', int, float, np_int, np_float, np_dt, pd_dt, np_td, pd_td)
DT: TypeAlias = Union[int, float, np_int, np_float, py_dt, np_dt, pd_dt]
TD: TypeAlias = Union[int, float, np_int, np_float, py_td, np_td, pd_td]
Time: TypeAlias = Union[DT, TD]
RealDTVar = TypeVar('RealDTVar', py_dt, np_dt, pd_dt)
RealTDVar = TypeVar('RealTDVar', py_td, np_td, pd_td)
RealTimeVar = TypeVar('RealTimeVar', py_dt, np_dt, pd_dt, py_td, np_td, pd_td)
RealDT: TypeAlias = Union[py_dt, np_dt, pd_dt]
RealTD: TypeAlias = Union[py_td, np_td, pd_td]
RealTime: TypeAlias = Union[RealDT, RealTD]
NumpyDTVar = TypeVar('NumpyDTVar', np_int, np_float, np_dt)
NumpyTDVar = TypeVar('NumpyTDVar', np_int, np_float, np_td)
NumpyTimeVar = TypeVar('NumpyTimeVar', np_int, np_float, np_dt, np_td)
