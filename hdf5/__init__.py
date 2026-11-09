"""Provide explicit, thread-safe access to HDF5 files."""

from ._hdf5 import AttributeManager as AttributeManager
from ._hdf5 import Dataset as Dataset
from ._hdf5 import File as File
from ._hdf5 import Group as Group
from ._hdf5 import StringInfo as StringInfo
from ._hdf5 import check_enum_dtype as check_enum_dtype
from ._hdf5 import check_opaque_dtype as check_opaque_dtype
from ._hdf5 import check_string_dtype as check_string_dtype
from ._hdf5 import check_vlen_dtype as check_vlen_dtype
from ._hdf5 import enum_dtype as enum_dtype
from ._hdf5 import opaque_dtype as opaque_dtype
from ._hdf5 import string_dtype as string_dtype
from ._hdf5 import vlen_dtype as vlen_dtype

__all__ = [
    "AttributeManager",
    "Dataset",
    "File",
    "Group",
    "StringInfo",
    "check_enum_dtype",
    "check_opaque_dtype",
    "check_string_dtype",
    "check_vlen_dtype",
    "enum_dtype",
    "opaque_dtype",
    "string_dtype",
    "vlen_dtype",
]
