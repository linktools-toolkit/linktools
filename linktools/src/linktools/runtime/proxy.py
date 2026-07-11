#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import functools
import math
import operator
import os
import sys
import typing as _t

if _t.TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.machinery import ModuleSpec

    P = _t.ParamSpec("P")
    T = _t.TypeVar("T")


def _default_cls_attr(name, type_, cls_value):

    def __new__(cls, getter):
        instance = type_.__new__(cls, cls_value)
        instance.__getter = getter
        return instance

    def __get__(self, obj, cls=None):
        return self.__getter(obj) if obj is not None else self

    return type(name, (type_,), {
        "__new__": __new__, "__get__": __get__,
    })


__module__ = __name__

_proxy_fn = "_Proxy__fn"
_proxy_object = "_Proxy__object"


class Proxy(object):

    __slots__ = ("__fn", "__object", "__dict__")
    __missing__ = object()

    def __init__(self, fn=__missing__, name=None, doc=None):
        object.__setattr__(self, _proxy_fn, fn)
        object.__setattr__(self, _proxy_object, Proxy.__missing__)
        if name is not None:
            object.__setattr__(self, "__custom_name__", name)
        if doc is not None:
            object.__setattr__(self, "__doc__", doc)

    @_default_cls_attr("name", str, __name__)
    def __name__(self):
        try:
            return self.__custom_name__
        except AttributeError:
            return self._get_current_object().__name__

    @_default_cls_attr("qualname", str, __name__)
    def __qualname__(self):
        try:
            return self.__custom_name__
        except AttributeError:
            return self._get_current_object().__qualname__

    @_default_cls_attr("module", str, __module__)
    def __module__(self):
        return self._get_current_object().__module__

    @_default_cls_attr("doc", str, __doc__)
    def __doc__(self):
        return self._get_current_object().__doc__

    def _get_class(self):
        return self._get_current_object().__class__

    @property
    def __class__(self):
        return self._get_class()

    def _get_current_object(self):
        obj = getattr(self, _proxy_object)
        if obj == Proxy.__missing__:
            obj = getattr(self, _proxy_fn)()
            object.__setattr__(self, _proxy_object, obj)
        return obj

    def _set_current_object(self, obj):
        object.__setattr__(self, _proxy_object, obj)
        return self

    @property
    def __dict__(self):
        return self._get_current_object().__dict__

    def __repr__(self):
        return repr(self._get_current_object())

    def __bool__(self):
        return bool(self._get_current_object())

    __nonzero__ = __bool__

    def __dir__(self):
        return dir(self._get_current_object())

    def __getattr__(self, name):
        if name == "__members__":
            return dir(self._get_current_object())
        return getattr(self._get_current_object(), name)

    def __setitem__(self, key, value):
        self._get_current_object()[key] = value

    def __delitem__(self, key):
        del self._get_current_object()[key]

    def __setslice__(self, i, j, seq):
        self._get_current_object()[i:j] = seq

    def __delslice__(self, i, j):
        del self._get_current_object()[i:j]

    def __setattr__(self, name, value):
        setattr(self._get_current_object(), name, value)

    def __delattr__(self, name):
        delattr(self._get_current_object(), name)

    def __str__(self):
        return str(self._get_current_object())

    def __bytes__(self):
        return bytes(self._get_current_object())

    def __format__(self, format_spec):
        return format(self._get_current_object(), format_spec)

    def __lt__(self, other):
        return self._get_current_object() < other

    def __le__(self, other):
        return self._get_current_object() <= other

    def __eq__(self, other):
        return self._get_current_object() == other

    def __ne__(self, other):
        return self._get_current_object() != other

    def __gt__(self, other):
        return self._get_current_object() > other

    def __ge__(self, other):
        return self._get_current_object() >= other

    def __hash__(self):
        return hash(self._get_current_object())

    def __call__(self, *a, **kw):
        return self._get_current_object()(*a, **kw)

    def __len__(self):
        return len(self._get_current_object())

    def __length_hint__(self):
        return self._get_current_object().__length_hint__()

    def __getitem__(self, i):
        return self._get_current_object()[i]

    def __iter__(self):
        return iter(self._get_current_object())

    def __next__(self):
        return next(self._get_current_object())

    def __reversed__(self):
        return reversed(self._get_current_object())

    def __contains__(self, i):
        return i in self._get_current_object()

    def __getslice__(self, i, j):
        return self._get_current_object()[i:j]

    def __add__(self, other):
        return self._get_current_object() + other

    def __sub__(self, other):
        return self._get_current_object() - other

    def __mul__(self, other):
        return self._get_current_object() * other

    def __matmul__(self, other):
        return self._get_current_object() @ other

    def __floordiv__(self, other):
        return self._get_current_object() // other

    def __mod__(self, other):
        return self._get_current_object() % other

    def __divmod__(self, other):
        return self._get_current_object().__divmod__(other)

    def __pow__(self, other):
        return self._get_current_object() ** other

    def __lshift__(self, other):
        return self._get_current_object() << other

    def __rshift__(self, other):
        return self._get_current_object() >> other

    def __and__(self, other):
        return self._get_current_object() & other

    def __xor__(self, other):
        return self._get_current_object() ^ other

    def __or__(self, other):
        return self._get_current_object() | other

    def __radd__(self, other):
        return other + self._get_current_object()

    def __rsub__(self, other):
        return other - self._get_current_object()

    def __rmul__(self, other):
        return other * self._get_current_object()

    def __rmatmul__(self, other):
        return other @ self._get_current_object()

    def __rfloordiv__(self, other):
        return other // self._get_current_object()

    def __rmod__(self, other):
        return other % self._get_current_object()

    def __rdivmod__(self, other):
        return divmod(other, self._get_current_object())

    def __rpow__(self, other):
        return other ** self._get_current_object()

    def __rlshift__(self, other):
        return other << self._get_current_object()

    def __rrshift__(self, other):
        return other >> self._get_current_object()

    def __rand__(self, other):
        return other & self._get_current_object()

    def __rxor__(self, other):
        return other ^ self._get_current_object()

    def __ror__(self, other):
        return other | self._get_current_object()

    def __iadd__(self, other):
        return self._set_current_object(operator.iadd(self._get_current_object(), other))

    def __isub__(self, other):
        return self._set_current_object(operator.isub(self._get_current_object(), other))

    def __imul__(self, other):
        return self._set_current_object(operator.imul(self._get_current_object(), other))

    def __imatmul__(self, other):
        return self._set_current_object(operator.imatmul(self._get_current_object(), other))

    def __ifloordiv__(self, other):
        return self._set_current_object(operator.ifloordiv(self._get_current_object(), other))

    def __imod__(self, other):
        return self._set_current_object(operator.imod(self._get_current_object(), other))

    def __ipow__(self, other):
        return self._set_current_object(operator.ipow(self._get_current_object(), other))

    def __ilshift__(self, other):
        return self._set_current_object(operator.ilshift(self._get_current_object(), other))

    def __irshift__(self, other):
        return self._set_current_object(operator.irshift(self._get_current_object(), other))

    def __iand__(self, other):
        return self._set_current_object(operator.iand(self._get_current_object(), other))

    def __ixor__(self, other):
        return self._set_current_object(operator.ixor(self._get_current_object(), other))

    def __ior__(self, other):
        return self._set_current_object(operator.ior(self._get_current_object(), other))

    def __div__(self, other):
        return self._get_current_object().__div__(other)

    def __truediv__(self, other):
        return self._get_current_object().__truediv__(other)

    def __rtruediv__(self, other):
        return other / self._get_current_object()

    def __itruediv__(self, other):
        return self._set_current_object(operator.itruediv(self._get_current_object(), other))

    def __neg__(self):
        return -(self._get_current_object())

    def __pos__(self):
        return +(self._get_current_object())

    def __abs__(self):
        return abs(self._get_current_object())

    def __invert__(self):
        return ~(self._get_current_object())

    def __complex__(self):
        return complex(self._get_current_object())

    def __int__(self):
        return int(self._get_current_object())

    def __float__(self):
        return float(self._get_current_object())

    def __oct__(self):
        return oct(self._get_current_object())

    def __hex__(self):
        return hex(self._get_current_object())

    def __index__(self):
        return self._get_current_object().__index__()

    def __round__(self, ndigits=None):
        obj = self._get_current_object()
        return round(obj) if ndigits is None else round(obj, ndigits)

    def __trunc__(self):
        return math.trunc(self._get_current_object())

    def __floor__(self):
        return math.floor(self._get_current_object())

    def __ceil__(self):
        return math.ceil(self._get_current_object())

    def __coerce__(self, other):
        return self._get_current_object().__coerce__(other)

    def __enter__(self):
        return self._get_current_object().__enter__()

    def __exit__(self, *a, **kw):
        return self._get_current_object().__exit__(*a, **kw)

    def __await__(self):
        return self._get_current_object().__await__()

    def __aiter__(self):
        return self._get_current_object().__aiter__()

    def __anext__(self):
        return self._get_current_object().__anext__()

    def __aenter__(self):
        return self._get_current_object().__aenter__()

    def __aexit__(self, *a, **kw):
        return self._get_current_object().__aexit__(*a, **kw)

    def __copy__(self):
        return copy.copy(self._get_current_object())

    def __deepcopy__(self, memo):
        return copy.deepcopy(self._get_current_object(), memo)

    def __reduce__(self):
        return self._get_current_object().__reduce__()

    def __reduce_ex__(self, protocol):
        return self._get_current_object().__reduce_ex__(protocol)


class IterProxy(_t.Iterable):

    __missing__ = object()

    def __init__(self, func: "_t.Callable[P, _t.Iterable[T]]", *args: "P.args", **kwargs: "P.kwargs"):
        self._data = IterProxy.__missing__
        self._fn = func
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        if self._data == IterProxy.__missing__:
            self._data = self._fn(*self._args, **self._kwargs)
        return iter(self._data)


def _load_lazy_module(name: str, spec: "ModuleSpec") -> "T":
    from importlib.util import LazyLoader, module_from_spec

    loader = LazyLoader(spec.loader)
    spec.loader = loader
    module = module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def import_module(name: str, spec: "ModuleSpec" = None) -> "T":
    from importlib.util import find_spec

    if name in sys.modules:
        return sys.modules[name]
    spec = spec or find_spec(name)
    if not spec:
        raise ModuleNotFoundError(f"No module named '{name}'")
    return _load_lazy_module(name, spec)


def import_module_file(name: str, path: str) -> "T":
    from importlib.util import spec_from_file_location

    if name in sys.modules:
        return sys.modules[name]
    if os.path.isdir(path):
        path = os.path.join(path, "__init__.py")
    if not os.path.exists(path):
        raise ModuleNotFoundError(f"No such file or directory: '{path}'")
    spec = spec_from_file_location(name, path)
    if not spec:
        raise ModuleNotFoundError(f"No module named '{name}'")
    return _load_lazy_module(name, spec)


def get_derived_type(t: "type[T]") -> "type[T]":

    class Derived(Proxy):

        def __init__(self, obj: "T"):
            super().__init__()
            object.__setattr__(self, "__super__", obj)

        def _get_current_object(self):
            return self.__super__

    return Derived


def lazy_load(fn: "Callable[P, T]", *args: "P.args", **kwargs: "P.kwargs") -> "T":
    return Proxy(functools.partial(fn, *args, **kwargs))


def raise_error(e: "BaseException"):
    raise e


def lazy_raise(e: "BaseException") -> "T":
    return lazy_load(raise_error, e)
