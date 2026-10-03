"""Test for the cpyutl package."""

import pytest

from cpyutl._cpyutl_test import test_nested_sequences as _test_nested_sequences


def test_nested_sequences():
    """Test test_nested_sequences function."""
    in_data = (
        # bool,
        True,
        # tuple[int, bool],
        (7, False),
        # tuple[float, float, object],
        (3.1, 35, print),
        # tuple[int, tuple[str, bool, tuple[float, int]], str]
        (-4, ("Hello", True, (3.14, 69)), "World"),
    )
    out_data = _test_nested_sequences(*in_data)
    assert out_data == in_data


def test_too_many_arguments_raise_type_error():
    """Test that more arguments than the spec declares is a TypeError.

    It used to be an assertion, which aborted the interpreter instead of
    raising: a caller's mistake should never take the process down.
    """
    with pytest.raises(TypeError) as info:
        _test_nested_sequences(True, (7, False), 3.1, "World", "one too many")
    assert "at most 4" in str(info.value.__cause__)

    with pytest.raises(TypeError):
        _test_nested_sequences(True, (7, False), 3.1, "World", extra=1)


if __name__ == "__main__":
    test_nested_sequences()
    test_too_many_arguments_raise_type_error()
