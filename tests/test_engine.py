"""Tests for tinylab.engine.use_calculator: the allow/deny surface the calculator tool-use state
machine relies on -- pure functions, no model or tokenizer needed."""
from tinylab.engine import use_calculator


def test_allows_pure_arithmetic():
    assert use_calculator("2 + 3 * 4") == 14


def test_disallows_power_operator():
    assert use_calculator("2 ** 10") is None


def test_allows_string_count():
    assert use_calculator("'mississippi'.count('s')") == 4


def test_disallows_dunder_access():
    assert use_calculator("().__class__.__bases__[0].__subclasses__()") is None


def test_disallows_import():
    assert use_calculator("__import__('os').listdir('.')") is None


def test_disallows_arbitrary_method_call_without_count():
    # allowed_chars would let this through, but the ".count(" gate should still block it
    assert use_calculator("'x'.upper()") is None


def test_strips_commas_from_numbers():
    assert use_calculator("1,000 + 1") == 1001


def test_unparseable_expression_returns_none_not_an_exception():
    assert use_calculator("1 +") is None
