# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib
from importlib.util import find_spec
from functools import lru_cache
from typing import Optional
import pkg_resources


@lru_cache()
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> package_available('os')
    True
    >>> package_available('bla')
    False

    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


@lru_cache()
def module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> module_available('os')
    True
    >>> module_available('os.bla')
    False
    >>> module_available('bla.bla')
    False

    """
    module_names = module_path.split(".")
    if not package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


class RequirementCache:
    """Boolean-like class to check for requirement and module availability.

    Args:
        requirement: The requirement to check, version specifiers are allowed.
        module: The optional module to try to import if the requirement check fails.

    >>> RequirementCache("torch>=0.1")
    Requirement 'torch>=0.1' met
    >>> bool(RequirementCache("torch>=0.1"))
    True
    >>> bool(RequirementCache("torch>100.0"))
    False
    >>> RequirementCache("torch")
    Requirement 'torch' met
    >>> bool(RequirementCache("torch"))
    True
    >>> bool(RequirementCache("unknown_package"))
    False
    >>> bool(RequirementCache(module="torch.utils"))
    True
    >>> bool(RequirementCache(module="unknown_package"))
    False
    >>> bool(RequirementCache(module="unknown.module.path"))
    False

    """

    def __init__(
        self, requirement: Optional[str] = None, module: Optional[str] = None
    ) -> None:
        if not (requirement or module):
            raise ValueError("At least one arguments need to be set.")
        self.requirement = requirement
        self.module = module

    def _check_requirement(self) -> None:
        assert self.requirement  # noqa: S101; needed for typing
        try:
            # first try the pkg_resources requirement
            pkg_resources.require(self.requirement)
            self.available = True
            self.message = f"Requirement {self.requirement!r} met"
        except Exception as ex:
            self.available = False
            self.message = f"{ex.__class__.__name__}: {ex}. HINT: Try running `pip install -U {self.requirement!r}`"
            req_include_version = any(c in self.requirement for c in "=<>")
            if not req_include_version or self.module is not None:
                module = (
                    self.requirement if self.module is None else self.module
                )
                # sometimes `pkg_resources.require()` fails but the module is importable
                self.available = module_available(module)
                if self.available:
                    self.message = f"Module {module!r} available"

    def _check_module(self) -> None:
        assert self.module  # noqa: S101; needed for typing
        self.available = module_available(self.module)
        if self.available:
            self.message = f"Module {self.module!r} available"
        else:
            self.message = f"Module not found: {self.module!r}. HINT: Try running `pip install -U {self.module}`"

    def _check_available(self) -> None:
        if hasattr(self, "available"):
            return
        if self.requirement:
            self._check_requirement()
        if getattr(self, "available", True) and self.module:
            self._check_module()

    def __bool__(self) -> bool:
        """Format as bool."""
        self._check_available()
        return self.available

    def __str__(self) -> str:
        """Format as string."""
        self._check_available()
        return self.message

    def __repr__(self) -> str:
        """Format as string."""
        return self.__str__()
