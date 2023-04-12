# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_strings():
    assert ak.type(ak.to_layout("hello")) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        1,
    )
    assert ak.type(ak.to_layout(["hello"])) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        1,
    )
    assert ak.type(ak.to_layout(b"hello")) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "byte"}),
            parameters={"__array__": "bytestring"},
        ),
        1,
    )
    assert ak.type(ak.to_layout([b"hello"])) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "byte"}),
            parameters={"__array__": "bytestring"},
        ),
        1,
    )


def test_python_scalars():
    assert ak.type(ak.to_layout(0)) == ak.types.ArrayType(
        ak.types.NumpyType("int64"), 1
    )
    assert ak.type(ak.to_layout(0.0)) == ak.types.ArrayType(
        ak.types.NumpyType("float64"), 1
    )
    assert ak.type(ak.to_layout(1j)) == ak.types.ArrayType(
        ak.types.NumpyType("complex128"), 1
    )
    assert ak.type(ak.to_layout(False)) == ak.types.ArrayType(
        ak.types.NumpyType("bool"), 1
    )


def test_numpy_scalars():
    assert ak.type(
        ak.to_layout(np.datetime64("2023-04-12T14:41:15"))
    ) == ak.types.ArrayType(ak.types.NumpyType("datetime64[s]"), 1)
    assert ak.type(ak.to_layout(np.timedelta64(100, "s"))) == ak.types.ArrayType(
        ak.types.NumpyType("timedelta64[s]"), 1
    )
    assert ak.type(ak.to_layout(np.array(10, dtype=np.float16))) == ak.types.ArrayType(
        ak.types.NumpyType("float16"), 1
    )
    assert ak.type(ak.to_layout(np.float16(10))) == ak.types.ArrayType(
        ak.types.NumpyType("float16"), 1
    )
    assert ak.type(ak.to_layout(np.array(10, dtype=np.int32))) == ak.types.ArrayType(
        ak.types.NumpyType("int32"), 1
    )
    assert ak.type(ak.to_layout(np.int32(10))) == ak.types.ArrayType(
        ak.types.NumpyType("int32"), 1
    )
    assert ak.type(ak.to_layout(np.array("hello", dtype="<U8"))) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        1,
    )
