# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import sys
from collections.abc import Mapping

import awkward as ak
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._regularize import is_integer
from awkward._typing import Any
from awkward.types.type import Type


class ArrayType:
    def __init__(
        self, content: Type, length: ShapeItem, behavior: Mapping | None = None
    ):
        if not isinstance(content, ak.types.Type):
            raise TypeError(
                "{} all 'contents' must be Type subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not ((is_integer(length) and length >= 0) or length is unknown_length):
            raise ValueError(
                "{} 'length' must be a non-negative integer or unknown length, not {}".format(
                    type(self).__name__, repr(length)
                )
            )
        self._content: Type = content
        self._length: ShapeItem = length
        self._behavior: Mapping | None = behavior

    @property
    def content(self) -> Type:
        return self._content

    @property
    def length(self) -> ShapeItem:
        return self._length

    @property
    def behavior(self) -> Mapping | None:
        return self._behavior

    def __str__(self) -> str:
        return "".join(self._str("", True))

    def show(self, stream=sys.stdout):
        stream.write("".join([*self._str("", False), "\n"]))

    def _str(self, indent: str, compact: bool) -> list[str]:
        return [
            f"{self._length} * ",
            *self._content._str(
                indent,
                compact,
                self._behavior,
            ),
        ]

    def __repr__(self) -> str:
        args = [repr(self._content), repr(self._length), repr(self._behavior)]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def is_equal_to(self, other: Any, *, all_parameters: bool = False) -> bool:
        return (
            isinstance(other, type(self))
            and (
                other._length is unknown_length
                or self._length is unknown_length
                or self._length == other._length
            )
            and self._content.is_equal_to(other._content, all_parameters=all_parameters)
        )

    __eq__ = is_equal_to
