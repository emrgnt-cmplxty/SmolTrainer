import os


def get_root_py_fpath() -> str:
    """Get the path to the root of the python code directory."""

    return os.path.dirname(os.path.realpath(__file__))


def get_root_fpath() -> str:
    """Get the path to the root of the baby_moe directory."""

    return os.path.join(get_root_py_fpath(), "..")
