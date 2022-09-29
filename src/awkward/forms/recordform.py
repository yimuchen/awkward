# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import glob
from collections.abc import Iterable
from typing import Any

import awkward as ak
from awkward.forms.form import Form, _parameters_equal
from awkward.forms.indexedform import IndexedForm


class RecordForm(Form):
    is_RecordType = True

    def __init__(
        self,
        contents,
        fields,
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(contents, Iterable):
            raise ak._util.error(
                TypeError(
                    "{} 'contents' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )
        for content in contents:
            if not isinstance(content, Form):
                raise ak._util.error(
                    TypeError(
                        "{} all 'contents' must be Form subclasses, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )
        if fields is not None and not isinstance(fields, Iterable):
            raise ak._util.error(
                TypeError(
                    "{} 'fields' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )

        self._fields = fields
        self._contents = list(contents)
        self._init(has_identifier, parameters, form_key)

    @property
    def fields(self):
        if self._fields is None:
            return [str(i) for i in range(len(self._contents))]
        else:
            return self._fields

    @property
    def is_tuple(self):
        return self._fields is None

    @property
    def contents(self):
        return self._contents

    def __repr__(self):
        args = [repr(self._contents), repr(self._fields)] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def index_to_field(self, index):
        if 0 <= index < len(self._contents):
            if self._fields is None:
                return str(index)
            else:
                return self._fields[index]
        else:
            raise ak._util.error(
                IndexError(
                    "no index {} in record with {} fields".format(
                        index, len(self._contents)
                    )
                )
            )

    def field_to_index(self, field):
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                pass
            else:
                if 0 <= i < len(self._contents):
                    return i
        else:
            try:
                i = self._fields.index(field)
            except ValueError:
                pass
            else:
                return i
        raise ak._util.error(
            IndexError(
                "no field {} in record with {} fields".format(
                    repr(field), len(self._contents)
                )
            )
        )

    def has_field(self, field):
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                return False
            else:
                return 0 <= i < len(self._contents)
        else:
            return field in self._fields

    def content(self, index_or_field):
        if ak._util.isint(index_or_field):
            index = index_or_field
        elif ak._util.isstr(index_or_field):
            index = self.field_to_index(index_or_field)
        else:
            raise ak._util.error(
                TypeError(
                    "index_or_field must be an integer (index) or string (field), not {}".format(
                        repr(index_or_field)
                    )
                )
            )
        return self._contents[index]

    def _tolist_part(self, verbose, toplevel):
        out = {"class": "RecordArray"}

        contents_tolist = [
            content._tolist_part(verbose, toplevel=False) for content in self._contents
        ]
        if self._fields is not None:
            out["contents"] = dict(zip(self._fields, contents_tolist))
        else:
            out["contents"] = contents_tolist

        return self._tolist_extra(out, verbose)

    def _type(self, typestrs):
        return ak.types.recordtype.RecordType(
            [x._type(typestrs) for x in self._contents],
            self._fields,
            self._parameters,
            ak._util.gettypestr(self._parameters, typestrs),
        )

    def __eq__(self, other):
        if isinstance(other, RecordForm):
            if (
                self._has_identifier == other._has_identifier
                and self._form_key == other._form_key
                and self.is_tuple == other.is_tuple
                and len(self._contents) == len(other._contents)
                and _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
            ):
                if self.is_tuple:
                    return self._contents == other._contents
                else:
                    return dict(zip(self._fields, self._contents)) == dict(
                        zip(other._fields, other._contents)
                    )
            else:
                return False
        else:
            return False

    def generated_compatibility(self, other):
        if other is None:
            return True

        elif isinstance(other, RecordForm):
            if self.is_tuple == other.is_tuple:
                self_fields = set(self._fields)
                other_fields = set(other._fields)
                if self_fields == other_fields:
                    return _parameters_equal(
                        self._parameters, other._parameters
                    ) and all(
                        self.content(x).generated_compatibility(other.content(x))
                        for x in self_fields
                    )
                else:
                    return False
            else:
                return False

        else:
            return False

    def _getitem_range(self):
        return RecordForm(
            self._contents,
            self._fields,
            has_identifier=self._has_identifier,
            parameters=self._parameters,
            form_key=None,
        )

    def _getitem_field(self, where, only_fields=()):
        if len(only_fields) == 0:
            return self.content(where)

        else:
            nexthead, nexttail = ak._slicing.headtail(only_fields)
            if ak._util.isstr(nexthead):
                return self.content(where)._getitem_field(nexthead, nexttail)
            else:
                return self.content(where)._getitem_fields(nexthead, nexttail)

    def _getitem_fields(self, where, only_fields=()):
        indexes = [self.field_to_index(field) for field in where]
        if self._fields is None:
            fields = None
        else:
            fields = [self._fields[i] for i in indexes]

        if len(only_fields) == 0:
            contents = [self.content(i) for i in indexes]
        else:
            nexthead, nexttail = ak._slicing.headtail(only_fields)
            if ak._util.isstr(nexthead):
                contents = [
                    self.content(i)._getitem_field(nexthead, nexttail) for i in indexes
                ]
            else:
                contents = [
                    self.content(i)._getitem_fields(nexthead, nexttail) for i in indexes
                ]

        return RecordForm(
            contents,
            fields,
            has_identifier=self._has_identifier,
            parameters=None,
            form_key=None,
        )

    def _carry(self, allow_lazy):
        if allow_lazy:
            return IndexedForm(
                "i64",
                self,
                has_identifier=self._has_identifier,
                parameters=None,
                form_key=None,
            )
        else:
            return RecordForm(
                self._contents,
                self._fields,
                has_identifier=self._has_identifier,
                parameters=self._parameters,
                form_key=None,
            )

    def purelist_parameter(self, key):
        return self.parameter(key)

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return 1

    @property
    def is_identity_like(self):
        return False

    @property
    def minmax_depth(self):
        if len(self._contents) == 0:
            return (1, 1)
        mins, maxs = [], []
        for content in self._contents:
            mindepth, maxdepth = content.minmax_depth
            mins.append(mindepth)
            maxs.append(maxdepth)
        return (min(mins), max(maxs))

    @property
    def branch_depth(self):
        if len(self._contents) == 0:
            return (False, 1)
        anybranch = False
        mindepth = None
        for content in self._contents:
            branch, depth = content.branch_depth
            if mindepth is None:
                mindepth = depth
            if branch or mindepth != depth:
                anybranch = True
            if mindepth > depth:
                mindepth = depth
        return (anybranch, mindepth)

    @property
    def dimension_optiontype(self):
        return False

    @property
    def dimension_parameters(self) -> dict[str, Any] | None:
        return self._parameters

    def _columns(self, path, output, list_indicator):
        for content, field in zip(self._contents, self.fields):
            content._columns(path + (field,), output, list_indicator)

    def _select_columns(self, index, specifier, matches, output):
        contents = []
        fields = []
        for content, field in zip(self._contents, self.fields):
            next_matches = [
                matches[i]
                and (index >= len(item) or glob.fnmatch.fnmatchcase(field, item[index]))
                for i, item in enumerate(specifier)
            ]
            if any(next_matches):
                len_output = len(output)
                next_content = content._select_columns(
                    index + 1, specifier, next_matches, output
                )
                if len_output != len(output):
                    contents.append(next_content)
                    fields.append(field)

        return RecordForm(
            contents,
            fields,
            self._has_identifier,
            self._parameters,
            self._form_key,
        )

    def _column_types(self):
        return sum((x._column_types() for x in self._contents), ())
