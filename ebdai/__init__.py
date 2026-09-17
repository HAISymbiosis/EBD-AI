"""BDI expansions that compose with ``ex_fuzzy``.

``pip install ebdai`` installs both packages. ``import ex_fuzzy`` is the
rebased Ex-Fuzzy API. ``import ebdai`` is for new BDI features that accept
``ex_fuzzy`` objects. Do not re-export Ex-Fuzzy names here: each class and
enum must exist once per process.
"""
import importlib

from ex_fuzzy._version import __version__

#: Register new expansion modules here. Tests and demos import through this
#: package, never the module files directly.
_SUBMODULES: tuple[str, ...] = ()

#: Top-level names and the submodule that defines each of them.
_EXPORTS: dict[str, str] = {}

__all__ = ['__version__', *_SUBMODULES, *_EXPORTS]


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', __name__)
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f'.{_EXPORTS[name]}', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))
