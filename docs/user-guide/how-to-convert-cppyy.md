---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

How to use Awkward Arrays in C++ with cppyy GIT
===============================================

The [cppyy](https://cppyy.readthedocs.io/en/latest/index.html) is an automatic, run-time, Python-C++ bindings generator, for calling C++ from Python and Python from C++. `cppyy` is based on the C++ interpreter `Cling`.

`cppyy` can understand Awkward Arrays. When an {class}`ak.Array` type is passed to a C++ function defined in `cppyy`, a `__cast_cpp__` magic function of an {class}`ak.Array` is invoked. The function dynamically generates a C++ type and a view of the array, if it has not been generated yet.

The view is a lightweight 40-byte C++ object dynamically allocated on the stack. This view is generated on demand - and only once per Awkward Array, the data are not copied.

```{code-cell} ipython3
import awkward as ak
import cppyy
```
Let's define an Awkward Array as a list of records:

```{code-cell} ipython3
array = ak.Array(
    [
        [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
        [],
        [{"x": 3, "y": [3.0, 0.3, 3.3]}],
    ]
)
```

This example shows a templated C++ function that takes an Awkward Array and iterates over the list of records:

```{code-cell} ipython3
source_code = """
template<typename T>
double go_fast_cpp(T& awkward_array) {
    double out = 0.0;

    for (auto list : awkward_array) {
        for (auto record : list) {
            for (auto item : record.y()) {
                out += item;
            }
        }
    }

    return out;
}
"""

cppyy.cppdef(source_code)
```

The C++ type of an Awkward Array is a made-up type;
`awkward::ListArray_ycQGTNoU57k`.

```{code-cell} ipython3
array.cpp_type
```

Awkward Arrays are dynamically typed, so in a C++ context, the type name is hashed. In practice, there is no need to know the type. The C++ code should use a placeholder type specifier `auto`. The type of the variable that is being declared will be automatically deduced from its initializer.

In a Python contexts, when a templated function requires a C++ type as a Python string, it can use the `ak.Array.cpp_type` property:

```{code-cell} ipython3
out = cppyy.gbl.go_fast_cpp[array.cpp_type](array)
assert out == ak.sum(array["y"])
```
