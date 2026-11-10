"""The spread test every studentised family in this package shares.

A studentised maximum divides each column's resampled deviation by a standard
error estimated from the same sample. Both are floating-point quantities whose
rounding error scales with the column's own magnitude, so a column of one
repeated value does not announce itself with a zero standard error unless that
value is exactly representable. These tests fix where the boundary between a
measurement and a rounding residue lies, and what happens when no column clears
it.
"""

import numpy as np
import pytest

from src.capability.core.statistics import columns_with_spread, studentisable_family


def _standard_errors(values: np.ndarray) -> np.ndarray:
    return np.nanstd(values, axis=0, ddof=1) / np.sqrt(len(values))


def test_a_column_whose_spread_is_only_rounding_carries_none():
    values = np.column_stack(
        [np.linspace(0.0, 1.0, 16), np.full(16, 0.3), np.zeros(16)]
    )
    errors = _standard_errors(values)
    # The two constant columns differ only in whether their value is exactly
    # representable, which is the whole of the defect: one standard error is
    # zero and the other is not, and neither column has any spread.
    assert errors[1] > 0.0 and errors[2] == 0.0
    assert columns_with_spread(values, errors).tolist() == [True, False, False]


def test_spread_is_judged_against_the_columns_own_magnitude():
    # Rescaling a column rescales its values and its standard error together, so
    # the verdict cannot depend on the units a contrast is reported in.
    values = np.column_stack([np.linspace(0.0, 1.0, 16), np.full(16, 0.3)])
    for scale in (1e-9, 1.0, 1e9):
        scaled = values * scale
        assert columns_with_spread(scaled, _standard_errors(scaled)).tolist() == [
            True,
            False,
        ]


def test_a_family_with_no_spread_anywhere_is_refused_where_that_is_declared():
    # The predicate answers only which columns carry spread. What an empty answer
    # means belongs to the statistic, not the family, so it lives in
    # `studentisable_family`: the native maximum reads the same family as a true
    # zero-width band, and the lanes that resample paired panels report it.
    values = np.full((16, 2), 0.3)
    errors = _standard_errors(values)
    assert columns_with_spread(values, errors).tolist() == [False, False]
    with pytest.raises(ValueError, match="no studentised maximum"):
        studentisable_family(values, errors)
    with pytest.raises(ValueError, match="non_identifiable"):
        studentisable_family(values, errors)
    # And where a column does carry spread the two agree exactly.
    mixed = np.column_stack([np.linspace(0.0, 1.0, 16), np.full(16, 0.3)])
    mixed_errors = _standard_errors(mixed)
    assert columns_with_spread(mixed, mixed_errors).tolist() == [True, False]
    assert studentisable_family(mixed, mixed_errors).tolist() == [True, False]


def test_the_table_and_the_standard_errors_must_describe_one_family():
    with pytest.raises(ValueError, match="one standard error per column"):
        columns_with_spread(np.zeros((4, 3)), np.ones(2))
    with pytest.raises(ValueError, match="units-by-columns table"):
        columns_with_spread(np.zeros(4), np.ones(4))
