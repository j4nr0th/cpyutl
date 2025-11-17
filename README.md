# `cpyutl` - CPython Utilities

The purpose of this project is to provide utilities for building Python C extensions.
It is intended to be used via `git-submodule` in C extensions built with CMake,
since they can then be added by simply using `add_subdirectory`, then linking the
`cpyutl` target.

## Features

Current features present:
- Parsing of arguments for functions Python functions/methods that specify `METH_FASTCALL` as one of their
  flags,
- Building of output values in the similar way to how the arguments are parsed in a more type safe way than `Py_BuildValue`,
- Checking NumPy array has the desired shape, flags, dtype, or any combination of these,
- Raising the exceptions when one is already set, with the original as the cause, the way `from e raise Exception` works in Python.
