# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable

import awkward as ak
from awkward._meta.bytemaskedmeta import ByteMaskedMeta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import type_parameters_equal
from awkward._typing import DType, Iterator, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form, _SpecifierMatcher, index_to_dtype

__all__ = ("ByteMaskedForm",)

np = NumpyMetadata.instance()


@final
class ByteMaskedForm(ByteMaskedMeta[Form], Form):
    _content: Form

    def __init__(
        self,
        mask,
        content,
        valid_when,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(mask, str):
            raise TypeError(
                f"{type(self).__name__} 'mask' must be of type str, not {mask!r}"
            )
        if not isinstance(content, Form):
            raise TypeError(
                "{} all 'contents' must be Form subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                "{} 'valid_when' must be bool, not {}".format(
                    type(self).__name__, repr(valid_when)
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._init(parameters=parameters, form_key=form_key)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    def copy(
        self,
        mask=UNSET,
        content=UNSET,
        valid_when=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return ByteMaskedForm(
            self._mask if mask is UNSET else mask,
            self._content if content is UNSET else content,
            self._valid_when if valid_when is UNSET else valid_when,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        mask,
        content,
        valid_when,
        *,
        parameters=None,
        form_key=None,
    ):
        if content.is_union:
            return content._union_of_optionarrays("i64", parameters)
        elif content.is_indexed or content.is_option:
            return ak.forms.IndexedOptionForm.simplified(
                "i64",
                content,
                parameters=parameters,
            )
        else:
            return cls(
                mask, content, valid_when, parameters=parameters, form_key=form_key
            )

    @property
    def is_identity_like(self) -> bool:
        return False

    def __repr__(self):
        args = [
            repr(self._mask),
            repr(self._content),
            repr(self._valid_when),
            *self._repr_args(),
        ]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "ByteMaskedArray",
                "mask": self._mask,
                "valid_when": self._valid_when,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.OptionType(
            self._content.type, parameters=self._parameters
        ).simplify_option_union()

    def __eq__(self, other):
        if isinstance(other, ByteMaskedForm):
            return (
                self._form_key == other._form_key
                and self._mask == other._mask
                and self._valid_when == other._valid_when
                and type_parameters_equal(self._parameters, other._parameters)
                and self._content == other._content
            )
        else:
            return False

    def _columns(self, path, output, list_indicator):
        self._content._columns(path, output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool) -> Self | None:
        next_content = self._content._prune_columns(is_inside_record_or_union)
        if next_content is None:
            return None
        else:
            return self.copy(content=next_content)

    def _select_columns(self, match_specifier: _SpecifierMatcher) -> Self:
        return self.copy(content=self._content._select_columns(match_specifier))

    def _column_types(self):
        return self._content._column_types()

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L206-L213
            has_identities, parameters, form_key, mask, content, valid_when = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(
                mask, content, valid_when, parameters=parameters, form_key=form_key
            )

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        yield (getkey(self, "mask"), index_to_dtype[self._mask])
        if recursive:
            yield from self._content._expected_from_buffers(getkey, recursive)
