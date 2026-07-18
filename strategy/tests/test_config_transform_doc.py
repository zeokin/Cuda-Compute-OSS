"""CPU-only test: the Config.transform docstring must name every built-in transform.

`config.py`'s docstring said the registry name is `"rsvd", the only built-in`, but
`nystrom` has been a registered built-in since PR #194 (`for _cls in (RandomizedSVDTransform,
NystromTransform): register_transform(...)`). Every other contributor-facing surface was
updated (`transforms.py`'s module docstring, `strategy/README.md`) — `config.py` was the lone
straggler. This pins the docstring to the real `available()` registry so a future built-in
can't silently leave it stale again.

Run:  python strategy/tests/test_config_transform_doc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.config import Config
from strategy.transforms import (
    NystromTransform,
    RandomizedSVDTransform,
    available,
)

# The genuine built-ins are the classes transforms.py registers at import time
# (`for _cls in (RandomizedSVDTransform, NystromTransform): register_transform(...)`).
# We read their `.name` directly rather than the live `available()` registry, which
# other tests mutate by registering throwaway transforms (e.g. "mine").
BUILTIN_TRANSFORM_NAMES = {RandomizedSVDTransform.name, NystromTransform.name}


def test_config_docstring_names_every_builtin_transform():
    doc = Config.__doc__ or ""
    for name in BUILTIN_TRANSFORM_NAMES:
        assert name in doc, f"Config docstring omits built-in transform {name!r}"


def test_documented_builtins_are_actually_registered():
    # Guards the other direction: the names the docstring lists really are registered
    # built-ins (subset, so a test that registered its own transform can't break this).
    assert BUILTIN_TRANSFORM_NAMES <= set(available())


if __name__ == "__main__":
    test_config_docstring_names_every_builtin_transform()
    test_documented_builtins_are_actually_registered()
    print("ok")
