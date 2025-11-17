"""Stub file for the C functions in cpyutl_test module."""

def test_nested_sequences(
    b1: bool,
    t1: tuple[int, bool],
    t2: tuple[float, float, object],
    t3: tuple[int, tuple[str, bool, tuple[float, int]], str],
) -> tuple[
    bool,
    tuple[int, bool],
    tuple[float, float, object],
    tuple[int, tuple[str, bool, tuple[float, int]], str],
]:
    """Returns its arguments, which are parsed as deeply nested sequences."""
    ...
