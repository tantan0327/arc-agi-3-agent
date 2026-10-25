"""Read the harness's intermediate_states.pkl without installing its dependencies.

Promoted from scratchpad/n2_load_states.py (2026-08-31) so the paper's §6
measurement is reproducible from the repo; logic unchanged.

`taaf/__init__` pulls in imageio and scipy at import time, neither of which this
project needs for anything else. Rather than stubbing dependencies one at a time,
this unpickles with a `find_class` that returns a transparent stand-in for any
class outside numpy/builtins, so the recorded structure can be walked directly.
"""
import pickle
import sys
import types


class Ghost:
    """Stands in for a class the pickle references. Keeps whatever state it is
    given so attribute names survive for inspection."""

    def __init__(self, *a, **k):
        self._args = a
        self._kw = k

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    # pickle uses these for objects rebuilt via __reduce__
    def append(self, x):
        self.__dict__.setdefault("_items", []).append(x)

    def extend(self, xs):
        self.__dict__.setdefault("_items", []).extend(xs)

    def __repr__(self):
        keys = [k for k in self.__dict__ if not k.startswith("_")]
        return f"<Ghost {keys[:10]}>"


def _ghost_factory(module, name):
    return type(f"Ghost_{name}", (Ghost,), {"__module__": module})


class Loader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("numpy", "builtins", "collections", "copyreg",
                              "datetime", "pathlib", "_codecs")):
            return super().find_class(module, name)
        try:
            return super().find_class(module, name)
        except Exception:  # noqa: BLE001
            return _ghost_factory(module, name)


def load(path):
    with open(path, "rb") as fh:
        return Loader(fh).load()


def describe(obj, depth=0, prefix="", max_depth=4):
    pad = "  " * depth
    if depth > max_depth:
        return
    t = type(obj).__name__
    if isinstance(obj, dict):
        print(f"{pad}{prefix}dict[{len(obj)}] keys={list(obj)[:6]}")
        for k in list(obj)[:2]:
            describe(obj[k], depth + 1, f"{k!r}: ")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{prefix}{t}[{len(obj)}]")
        if obj:
            describe(obj[0], depth + 1, "[0] ")
    elif hasattr(obj, "shape"):
        print(f"{pad}{prefix}array{getattr(obj, 'shape', '')} dtype={getattr(obj,'dtype','')}")
    elif isinstance(obj, Ghost):
        keys = [k for k in obj.__dict__ if not k.startswith("_")]
        print(f"{pad}{prefix}{t} attrs={keys[:12]}")
        for k in keys[:3]:
            describe(getattr(obj, k), depth + 1, f".{k} ")
    else:
        print(f"{pad}{prefix}{t} {str(obj)[:70]}")


if __name__ == "__main__":
    d = load(sys.argv[1] if len(sys.argv) > 1
             else "scratchpad/kout-is/intermediate_states.pkl")
    describe(d)
