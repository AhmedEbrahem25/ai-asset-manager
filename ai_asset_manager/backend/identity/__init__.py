r"""Working out what an anonymously named asset actually is.

A whole-machine scan finds a great many models called ``model``. Applications ship them
that way — ``model.tflite``, ``model.onnx``, ``model.pb`` — because inside the application
there is only one, so it needs no name. Catalogued verbatim, fifty of them produce fifty
identical rows and the inventory is useless exactly where it should be most interesting.

The information is not missing, though. It is in the path. ``AppData\\Local\\Google\\
Chrome\\User Data\\screen_ai\\...\\model.tflite`` says vendor, product, component and task
in that order, and a person reading it has no trouble at all. This package encodes what
that person knows.

Two modules: :mod:`~ai_asset_manager.backend.identity.vendors` holds the tables — who ships
what, which words name a task — and :mod:`~ai_asset_manager.backend.identity.naming` holds
the derivation. Neither touches the filesystem: identity is derived from a path string and
the facts already parsed, so it costs nothing at scan time and can be recomputed at any
point from what the catalogue stored.
"""

from __future__ import annotations

from ai_asset_manager.backend.identity.naming import (
    AssetIdentity,
    identify,
    is_generic_name,
)

__all__ = ["AssetIdentity", "identify", "is_generic_name"]
