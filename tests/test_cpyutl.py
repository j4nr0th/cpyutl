"""Test for the cpyutl package."""

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


if __name__ == "__main__":
    test_nested_sequences()
