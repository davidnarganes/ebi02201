import pytest

from ebi02201.utils import add_two

def test_add_two():
    """Test that the function adds two as indended."""
    assert add_two(3) == 5


def test_add_two_wrong_input():
    """Check that the function raises a type error for the wrong input type."""
    with pytest.raises(TypeError):
        add_two("blue")