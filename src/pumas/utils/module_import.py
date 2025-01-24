from contextlib import contextmanager
from typing import Any, Callable, Optional


@contextmanager
def switch_library(
    module: Any, library_type: str, switcher: Callable[[str], Any], new_library: str
):
    original_library: Optional[Any] = getattr(module, library_type, None)
    try:
        setattr(module, library_type, switcher(new_library))
        yield
    finally:
        if original_library is not None:
            setattr(module, library_type, original_library)
