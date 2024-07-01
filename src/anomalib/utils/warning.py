"""Warning util functions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TypeVar, cast

T = TypeVar("T")


def create_class_alias_with_deprecation_warning(new_class: type[T], deprecated_name: str) -> type[T]:
    """Creates a class alias with a deprecation warning.

    This function creates a new class that inherits from the provided class
    and issues a deprecation warning when instantiated. It's useful for
    maintaining backward compatibility while encouraging the use of a new class name.

    Args:
        new_class (Type[T]): The class that should be used going forward.
        deprecated_name (str): The old name of the class that is being deprecated.

    Returns:
        Type[T]: A new class that acts as a deprecated alias for `new_class`.

    Example:
        >>> class NewClass:
        ...     def __init__(self, value):
        ...         self.value = value
        >>> OldClass = create_class_with_deprecation_warning(NewClass, "OldClass")
        >>> instance = OldClass(42)  # This will issue a deprecation warning
    """

    def deprecated_init(*args, **kwargs) -> T:
        warnings.warn(
            f"{deprecated_name} is deprecated. Use {new_class.__name__} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_class(*args, **kwargs)

    return cast(type[T], type(deprecated_name, (new_class,), {"__init__": deprecated_init}))
