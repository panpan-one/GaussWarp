__all__ = ['ScalarDType', 'NUMPY_DTYPES', 'TORCH_DTYPES', 'PANDAS_DTYPES', 'PYTHON_DTYPES', 'NUMPY_TYPESTRINGS', 'NUMPY_TYPECODES', 'TORCH_TYPESTRINGS', 'PANDAS_TYPESTRINGS', 'PYTHON_TYPESTRINGS', 'NUMPY_BOOL_TYPECODES', 'NUMPY_COMPLEX_TYPECODES', 'NUMPY_FLOAT_TYPECODES', 'NUMPY_INT_TYPECODES', 'NUMPY_OBJECT_TYPECODES', 'NUMPY_STRING_TYPECODES', 'NUMPY_TIME_TYPECODES', 'NUMPY_UINT_TYPECODES', 'NUMPY_BOOL_TYPESTRINGS', 'NUMPY_COMPLEX_TYPESTRINGS', 'NUMPY_FLOAT_TYPESTRINGS', 'NUMPY_INT_TYPESTRINGS', 'NUMPY_OBJECT_TYPESTRINGS', 'NUMPY_STRING_TYPESTRINGS', 'NUMPY_TIME_TYPESTRINGS', 'NUMPY_UINT_TYPESTRINGS', 'TORCH_BOOL_TYPESTRINGS', 'TORCH_COMPLEX_TYPESTRINGS', 'TORCH_FLOAT_TYPESTRINGS', 'TORCH_INT_TYPESTRINGS', 'TORCH_UINT_TYPESTRINGS', 'BOOLS', 'CATEGORIES', 'PRECISION', 'EMOJIS', 'STRINGS', 'TYPESTRINGS']
from collections import namedtuple
from datetime import datetime, timedelta
from typing import Final, TypeAlias
import numpy as np
import pandas
import torch
from pandas.api.extensions import ExtensionDtype
NUMPY_INT_TYPECODES: Final[dict[type[np.signedinteger], str]] = {np.int8: 'b', np.int16: 'h', np.int32: 'i', np.int64: 'l', np.byte: 'b', np.short: 'h', np.intc: 'i', np.int_: 'l', np.intp: 'l', np.longlong: 'q'}
NUMPY_UINT_TYPECODES: Final[dict[type[np.unsignedinteger], str]] = {np.uint8: 'B', np.uint16: 'H', np.uint32: 'I', np.uint64: 'L', np.ubyte: 'B', np.ushort: 'H', np.uintc: 'I', np.uint: 'L', np.uintp: 'L', np.ulonglong: 'Q'}
NUMPY_FLOAT_TYPECODES: Final[dict[type[np.floating], str]] = {np.float16: 'e', np.float32: 'f', np.float64: 'd', np.float128: 'g', np.half: 'e', np.single: 'f', np.double: 'd', np.longdouble: 'g'}
NUMPY_COMPLEX_TYPECODES: Final[dict[type[np.complexfloating], str]] = {np.complex64: 'F', np.complex128: 'D', np.complex256: 'G', np.csingle: 'F', np.cdouble: 'D', np.clongdouble: 'G'}
NUMPY_TIME_TYPECODES: Final[dict[type[np.generic], str]] = {np.timedelta64: 'M', np.datetime64: 'm'}
NUMPY_BOOL_TYPECODES: Final[dict[type[np.generic], str]] = {np.bool_: '?'}
NUMPY_STRING_TYPECODES: Final[dict[type[np.flexible], str]] = {np.bytes_: 'S', np.str_: 'U', np.void: 'V'}
NUMPY_OBJECT_TYPECODES: Final[dict[type[np.generic], str]] = {np.object_: 'O'}
NUMPY_TYPECODES: Final[dict[type[np.generic], str]] = NUMPY_INT_TYPECODES | NUMPY_UINT_TYPECODES | NUMPY_FLOAT_TYPECODES | NUMPY_COMPLEX_TYPECODES | NUMPY_TIME_TYPECODES | NUMPY_STRING_TYPECODES | NUMPY_OBJECT_TYPECODES
NUMPY_INT_TYPESTRINGS: Final[dict[type[np.signedinteger], str]] = {np.int8: 'int8', np.int16: 'int16', np.int32: 'int32', np.int64: 'int64', np.byte: 'int8', np.short: 'int16', np.intc: 'int32', np.int_: 'int64', np.intp: 'int64', np.longlong: 'q'}
NUMPY_UINT_TYPESTRINGS: Final[dict[type[np.unsignedinteger], str]] = {np.uint8: 'uint8', np.uint16: 'uint16', np.uint32: 'uint32', np.uint64: 'uint64', np.ubyte: 'uint8', np.ushort: 'uint16', np.uintc: 'uint32', np.uint: 'uint64', np.uintp: 'uint64', np.ulonglong: 'Q'}
NUMPY_FLOAT_TYPESTRINGS: Final[dict[type[np.floating], str]] = {np.float16: 'float16', np.float32: 'float32', np.float64: 'float64', np.float128: 'float128', np.half: 'float16', np.single: 'float32', np.double: 'float64', np.longdouble: 'float128'}
NUMPY_COMPLEX_TYPESTRINGS: Final[dict[type[np.complexfloating], str]] = {np.complex64: 'complex64', np.complex128: 'complex128', np.complex256: 'complex256', np.csingle: 'complex64', np.cdouble: 'complex128', np.clongdouble: 'complex256'}
NUMPY_TIME_TYPESTRINGS: Final[dict[type[np.generic], str]] = {np.timedelta64: 'timedelta64', np.datetime64: 'datetime64'}
NUMPY_BOOL_TYPESTRINGS: Final[dict[type[np.generic], str]] = {np.bool_: 'bool'}
NUMPY_STRING_TYPESTRINGS: Final[dict[type[np.flexible], str]] = {np.bytes_: 'bytes', np.str_: 'str', np.void: 'void'}
NUMPY_OBJECT_TYPESTRINGS: Final[dict[type[np.generic], str]] = {np.object_: 'object'}
NUMPY_TYPESTRINGS: Final[dict[type[np.generic], str]] = NUMPY_INT_TYPESTRINGS | NUMPY_UINT_TYPESTRINGS | NUMPY_FLOAT_TYPESTRINGS | NUMPY_COMPLEX_TYPESTRINGS | NUMPY_TIME_TYPESTRINGS | NUMPY_STRING_TYPESTRINGS | NUMPY_OBJECT_TYPESTRINGS
NUMPY_DTYPES: Final[dict[str, type[np.generic]]] = {'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64, 'float16': np.float16, 'float32': np.float32, 'float64': np.float64, 'float128': np.float128, 'complex64': np.complex64, 'complex128': np.complex128, 'complex256': np.complex256, 'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64, 'timedelta64': np.timedelta64, 'datetime64': np.datetime64, 'bool': np.bool_, 'bytes': np.bytes_, 'str': np.str_, 'void': np.void, 'object': np.object_}
PANDAS_TYPESTRINGS: Final[dict[type[ExtensionDtype], str]] = {pandas.BooleanDtype: 'boolean', pandas.CategoricalDtype: 'category', pandas.DatetimeTZDtype: 'datetime64[ns, tz]', pandas.Float32Dtype: 'Float32', pandas.Float64Dtype: 'Float64', pandas.Int16Dtype: 'Int16', pandas.Int32Dtype: 'Int32', pandas.Int64Dtype: 'Int64', pandas.Int8Dtype: 'Int8', pandas.IntervalDtype: 'interval', pandas.PeriodDtype: 'period', pandas.SparseDtype: 'Sparse', pandas.StringDtype: 'string', pandas.UInt16Dtype: 'UInt16', pandas.UInt32Dtype: 'UInt32', pandas.UInt64Dtype: 'UInt64', pandas.UInt8Dtype: 'UInt8'}
PANDAS_DTYPES: Final[dict[str, type[ExtensionDtype]]] = {'boolean': pandas.BooleanDtype, 'category': pandas.CategoricalDtype, 'datetime64[ns, tz]': pandas.DatetimeTZDtype, 'Float32': pandas.Float32Dtype, 'Float64': pandas.Float64Dtype, 'Int16': pandas.Int16Dtype, 'Int32': pandas.Int32Dtype, 'Int64': pandas.Int64Dtype, 'Int8': pandas.Int8Dtype, 'interval': pandas.IntervalDtype, 'period': pandas.PeriodDtype, 'Sparse': pandas.SparseDtype, 'string': pandas.StringDtype, 'UInt16': pandas.UInt16Dtype, 'UInt32': pandas.UInt32Dtype, 'UInt64': pandas.UInt64Dtype, 'UInt8': pandas.UInt8Dtype}
TORCH_INT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {torch.int8: 'int8', torch.int16: 'int16', torch.int32: 'int32', torch.int64: 'int64', torch.qint8: 'qint8', torch.qint32: 'qint32'}
TORCH_UINT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {torch.uint8: 'uint8', torch.quint8: 'quint8', torch.quint4x2: 'quint4x2'}
TORCH_FLOAT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {torch.bfloat16: 'bfloat16', torch.float16: 'float16', torch.float32: 'float32', torch.float64: 'float64'}
TORCH_COMPLEX_TYPESTRINGS: Final[dict[torch.dtype, str]] = {torch.complex32: 'complex32', torch.complex64: 'complex64', torch.complex128: 'complex128'}
TORCH_BOOL_TYPESTRINGS: Final[dict[torch.dtype, str]] = {torch.bool: 'bool'}
TORCH_TYPESTRINGS: Final[dict[torch.dtype, str]] = TORCH_INT_TYPESTRINGS | TORCH_UINT_TYPESTRINGS | TORCH_FLOAT_TYPESTRINGS | TORCH_COMPLEX_TYPESTRINGS | TORCH_BOOL_TYPESTRINGS
TORCH_DTYPES: Final[dict[str, torch.dtype]] = {'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64, 'qint8': torch.qint8, 'qint32': torch.qint32, 'uint8': torch.uint8, 'quint8': torch.quint8, 'quint4x2': torch.quint4x2, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64, 'complex32': torch.complex32, 'complex64': torch.complex64, 'complex128': torch.complex128, 'bool': torch.bool}
PYTHON_DTYPES: Final[dict[str, type]] = {'bool': bool, 'int': int, 'float': float, 'complex': complex, 'str': str, 'bytes': bytes, 'datetime': datetime, 'timedelta': timedelta, 'object': object}
PYTHON_TYPESTRINGS: Final[dict[type, str]] = {bool: 'bool', int: 'int', float: 'float', complex: 'complex', str: 'str', bytes: 'bytes', datetime: 'datetime', timedelta: 'timedelta', object: 'object'}
ScalarDType: TypeAlias = type[np.generic] | torch.dtype | type[ExtensionDtype]
TYPESTRINGS: Final[dict[type[np.generic] | torch.dtype | type[ExtensionDtype], str]] = NUMPY_TYPESTRINGS | TORCH_TYPESTRINGS | PANDAS_TYPESTRINGS
PRECISION: Final[dict] = {16: 2 ** (-11), 32: 2 ** (-24), 64: 2 ** (-53), torch.float16: 2 ** (-11), torch.float32: 2 ** (-24), torch.float64: 2 ** (-53), np.float16: 2 ** (-11), np.float32: 2 ** (-24), np.float64: 2 ** (-53)}
BOOLS: Final[list[bool]] = [True, False]
EMOJIS: Final[list[str]] = list('😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟😠😡😢😣😤😥😦😧😨😩😪😫😬😭😮😯😰😱😲😳😴😵😶😷😸😹😺😻😼😽😾😿🙀🙁🙂🙃🙄🙅🙆🙇🙈🙉🙊🙋🙌🙍🙎🙏')
STRINGS: Final[list[str]] = ['Alfa', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliett', 'Kilo', 'Lima', 'Mike', 'November', 'Oscar', 'Papa', 'Quebec', 'Romeo', 'Sierra', 'Tango', 'Uniform', 'Victor', 'Whiskey', 'X-ray', 'Yankee', 'Zulu']
label = namedtuple('label', ['object', 'color'])
CATEGORIES: Final[list[label]] = [label(object='bear', color='brown'), label(object='bear', color='black'), label(object='bear', color='white'), label(object='beet', color='red'), label(object='beet', color='yellow'), label(object='beet', color='orange'), label(object='beet', color='white'), label(object='beet', color='violet')]
