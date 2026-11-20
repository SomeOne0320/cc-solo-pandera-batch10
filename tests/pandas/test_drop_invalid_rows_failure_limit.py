"""Failure-case reporting limits must not limit invalid-row removal."""

import pandas as pd
import pytest

import pandera.pandas as pa


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize("element_wise", [False, True])
@pytest.mark.parametrize(
    "values", [[], [-1.0, -2.0], [-1.0, 2.0, -3.0], [1.0, 2.0]]
)
@pytest.mark.parametrize(
    "schema_type", ["dataframe", "series", "column", "nested"]
)
def test_drop_invalid_rows_with_failure_limit(
    n_failure_cases, element_wise, values, schema_type
):
    """Column and series checks remove every failing row regardless of cap."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=float)})
    check = pa.Check(
        lambda x: x > 0,
        element_wise=element_wise,
        n_failure_cases=n_failure_cases,
    )
    expected = frame.loc[frame["x"] > 0]
    data = frame
    if schema_type == "dataframe":
        schema = pa.DataFrameSchema(
            {"x": pa.Column(float, check)}, drop_invalid_rows=True
        )
    elif schema_type == "series":
        schema = pa.SeriesSchema(float, check, drop_invalid_rows=True)
        data, expected = frame["x"], expected["x"]
    elif schema_type == "column":
        schema = pa.Column(float, check, name="x", drop_invalid_rows=True)
    else:
        schema = pa.DataFrameSchema(
            {"x": pa.Column(float, check, drop_invalid_rows=True)}
        )

    validated = schema.validate(data, lazy=True)

    if schema_type == "series":
        pd.testing.assert_series_equal(validated, expected)
    else:
        pd.testing.assert_frame_equal(validated, expected)


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize(
    "output_type", ["series", "dataframe", "element_wise"]
)
@pytest.mark.parametrize("values", [[], [-1, -2], [-1, 2, -3], [1, 2]])
def test_drop_invalid_rows_dataframe_check_with_failure_limit(
    n_failure_cases, output_type, values
):
    """Dataframe check masks remain complete when diagnostics are truncated."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype="int64")})
    check_fn = (
        (lambda df: df > 0)
        if output_type == "dataframe"
        else (lambda df: df["x"] > 0)
    )
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int)},
        checks=pa.Check(
            check_fn,
            element_wise=output_type == "element_wise",
            n_failure_cases=n_failure_cases,
        ),
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.loc[frame["x"] > 0])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_with_failure_limit_across_columns(n_failure_cases):
    """A dataframe mask removes rows failing in any checked column."""
    frame = pd.DataFrame(
        {"x": [-1, 2, 3], "y": [4, -5, 6]}, index=["first", "second", "third"]
    )
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int), "y": pa.Column(int)},
        checks=pa.Check(lambda df: df > 0, n_failure_cases=n_failure_cases),
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.loc[["third"]])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_failure_limit_still_caps_diagnostics(n_failure_cases):
    """Using the full row mask for dropping does not expand error reports."""
    frame = pd.DataFrame({"x": [-1, -2, 3]})
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))}
    )

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(frame, lazy=True)

    assert len(exc_info.value.failure_cases) == (
        2 if n_failure_cases is None else n_failure_cases
    )


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_with_failure_limit_requires_lazy(n_failure_cases):
    """The reporting cap does not change the lazy-validation requirement."""
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    with pytest.raises(pa.errors.SchemaDefinitionError, match="lazy"):
        schema.validate(pd.DataFrame({"x": [-1, 2]}))


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("output_type", ["column", "dataframe"])
def test_drop_invalid_rows_failure_limit_preserves_ignored_nulls(
    n_failure_cases, ignore_na, output_type
):
    """Only rows skipped by ignore_na survive a failed check."""
    frame = pd.DataFrame({"x": [-1.0, None, -2.0, 3.0]})

    def make_schema(limit):
        check = pa.Check(
            lambda obj: obj > 0, n_failure_cases=limit, ignore_na=ignore_na
        )
        if output_type == "dataframe":
            return pa.DataFrameSchema(
                {"x": pa.Column(float, nullable=True)},
                checks=check,
                drop_invalid_rows=True,
            )
        return pa.DataFrameSchema(
            {"x": pa.Column(float, check, nullable=True)},
            drop_invalid_rows=True,
        )

    schema = make_schema(n_failure_cases)

    validated = schema.validate(frame, lazy=True)

    expected = frame.loc[[1, 3]] if ignore_na else frame.loc[[3]]
    pd.testing.assert_frame_equal(validated, expected)


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_failure_limit_keeps_warning_rows(n_failure_cases):
    """A warning check still does not cause rows to be removed."""
    frame = pd.DataFrame({"x": [-1, -2, 3]})
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                pa.Check.gt(
                    0, n_failure_cases=n_failure_cases, raise_warning=True
                ),
            )
        },
        drop_invalid_rows=True,
    )

    with pytest.warns(pa.errors.SchemaWarning):
        validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame)


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize(
    "index",
    [
        pd.Index(["bad1", "bad2", "good"], name="row"),
        pd.date_range("2026-01-01", periods=3, name="row"),
        pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 1)], names=["group", "row"]
        ),
    ],
)
def test_drop_invalid_rows_failure_limit_preserves_index(
    n_failure_cases, index
):
    """Full masks preserve each supported unique index's type and names."""
    frame = pd.DataFrame({"x": [-1, -2, 3]}, index=index)
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[2]])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_failure_limit_multiple_checks(n_failure_cases):
    """Every limited check contributes its full set of failing rows."""
    frame = pd.DataFrame({"x": [-1, -2, 1, 4, 5]})
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                [
                    pa.Check.gt(0, n_failure_cases=n_failure_cases),
                    pa.Check.lt(3, n_failure_cases=n_failure_cases),
                ],
            )
        },
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[2]])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_failure_limit_respects_subsample(n_failure_cases):
    """A reporting cap does not extend validation to unsampled rows."""
    frame = pd.DataFrame({"x": [-1, -2, -3, 4]})
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, head=2, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[2:])


def _table_check_failure_cases(frame, n_failure_cases):
    """Validate with a dataframe-level check and return its failure cases."""
    schema = pa.DataFrameSchema(
        checks=pa.Check(lambda df: df > 0, n_failure_cases=n_failure_cases)
    )
    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return exc.failure_cases
    raise AssertionError("expected the table-level check to fail")


@pytest.mark.parametrize("n_failure_cases", [0, 1, 2, None])
@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(3),
        pd.Index([10, 20, 30]),
        pd.Index(["a", "b", "c"]),
        pd.Index(["a", "b", "c"], name="row"),
        pd.date_range("2026-01-01", periods=3),
    ],
)
def test_table_check_failure_limit_preserves_labels(n_failure_cases, index):
    """A table-level report limit keeps each row's original labels and values."""
    frame = pd.DataFrame(
        {"x": [-1.0, 2.0, -3.0], "y": [4.0, -5.0, 6.0]}, index=index
    )

    failure_cases = _table_check_failure_cases(frame, n_failure_cases)

    expected_labels = {
        None: [index[0], index[1], index[2]],
        0: [],
        1: [index[0]],
        2: [index[0], index[1]],
    }[n_failure_cases]
    expected_values = {
        None: [{"x": -1.0}, {"y": -5.0}, {"x": -3.0}],
        0: [],
        1: [{"x": -1.0}],
        2: [{"x": -1.0}, {"y": -5.0}],
    }[n_failure_cases]
    assert list(failure_cases["index"]) == expected_labels
    assert list(failure_cases["failure_case"]) == expected_values
    assert failure_cases["index"].dtype == frame.index.dtype


@pytest.mark.parametrize("n_failure_cases", [0, 1, 2, None])
def test_table_check_failure_limit_matches_full_report_prefix(
    n_failure_cases,
):
    """Capping the report only removes trailing rows from the full report."""
    frame = pd.DataFrame(
        {"x": [-1.0, 2.0, -3.0], "y": [4.0, -5.0, 6.0]},
        index=["a", "b", "c"],
    )

    full = _table_check_failure_cases(frame, None)
    limited = _table_check_failure_cases(frame, n_failure_cases)

    if n_failure_cases is None:
        pd.testing.assert_frame_equal(limited, full)
    else:
        # An empty capped report keeps the legacy empty-report dtype.
        reference = full.head(n_failure_cases)
        assert list(limited["index"]) == list(reference["index"])
        assert list(limited["failure_case"]) == list(reference["failure_case"])


@pytest.mark.parametrize("n_failure_cases", [1, 2, None])
def test_table_check_failure_limit_merges_duplicate_row_labels(
    n_failure_cases,
):
    """Rows sharing a label keep reporting the same merged payload."""
    frame = pd.DataFrame(
        {"x": [-1.0, 2.0, -3.0], "y": [4.0, -5.0, 6.0]},
        index=pd.Index([3, 3, 4], name="row"),
    )

    failure_cases = _table_check_failure_cases(frame, n_failure_cases)

    if n_failure_cases is None:
        assert list(failure_cases["index"]) == [3, 4]
        assert list(failure_cases["failure_case"]) == [
            {"x": -1.0, "y": -5.0},
            {"x": -3.0},
        ]
    else:
        assert list(failure_cases["index"]) == [3, 4][:n_failure_cases]
        assert (
            list(failure_cases["failure_case"])
            == [
                {"x": -1.0, "y": -5.0},
                {"x": -3.0},
            ][:n_failure_cases]
        )


@pytest.mark.parametrize("n_failure_cases", [0, 1, 2, None])
def test_table_check_failure_limit_handles_nulls_and_partial_failures(
    n_failure_cases,
):
    """Null cells are skipped and only failing rows are reported."""
    frame = pd.DataFrame(
        {
            "x": [-1.0, None, -3.0, 4.0],
            "y": [5.0, -6.0, None, 8.0],
        },
        index=pd.Index([1.0, 2.0, 3.0, 4.0]),
    )

    failure_cases = _table_check_failure_cases(frame, n_failure_cases)

    expected = (
        [1.0, 2.0, 3.0]
        if n_failure_cases is None
        else [
            1.0,
            2.0,
            3.0,
        ][:n_failure_cases]
    )
    assert list(failure_cases["index"]) == expected
    assert (
        list(failure_cases["failure_case"])
        == [
            {"x": -1.0},
            {"y": -6.0},
            {"x": -3.0},
        ][: len(expected)]
    )


@pytest.mark.parametrize("n_failure_cases", [1, 2, 3, None])
def test_table_check_failure_limit_deduplicates_reports(n_failure_cases):
    """Duplicate payloads are reported once, before the cap is applied."""
    frame = pd.DataFrame(
        {"x": [-1.0, 2.0, -1.0], "y": [3.0, -4.0, 3.0]},
        index=["a", "b", "c"],
    )

    failure_cases = _table_check_failure_cases(frame, n_failure_cases)

    if n_failure_cases is None:
        # Without a cap the report keeps every failing row.
        assert list(failure_cases["index"]) == ["a", "b", "c"]
        assert list(failure_cases["failure_case"]) == [
            {"x": -1.0},
            {"y": -4.0},
            {"x": -1.0},
        ]
    else:
        expected = [{"x": -1.0}, {"y": -4.0}][:n_failure_cases]
        assert list(failure_cases["failure_case"]) == expected


def test_table_check_failure_limit_zero_returns_empty_report():
    """A zero cap returns the same empty report shape as before."""
    frame = pd.DataFrame({"x": [-1.0, 2.0], "y": [3.0, -4.0]})

    failure_cases = _table_check_failure_cases(frame, 0)

    assert list(failure_cases.columns) == [
        "schema_context",
        "column",
        "check",
        "check_number",
        "failure_case",
        "index",
    ]
    assert failure_cases.empty
    assert failure_cases["index"].dtype == frame.index.dtype


def test_table_check_failure_limit_does_not_change_pass_result():
    """The table-level pass result still reflects every cell, not the cap."""
    frame = pd.DataFrame({"x": [-1.0, 2.0], "y": [3.0, -4.0]})
    schema = pa.DataFrameSchema(
        checks=pa.Check(lambda df: df > 0, n_failure_cases=1)
    )

    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(frame, lazy=True)

    # The cap does not affect whether the schema passes: an all-passing frame
    # validates regardless of how many failure cases would have been reported.
    passing = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    pd.testing.assert_frame_equal(schema.validate(passing, lazy=True), passing)


def _table_check_peak_bytes(limit, n_rows=5000, n_cols=8):
    """Peak tracked allocations while validating with a table-level check."""
    import gc
    import tracemalloc

    frame = pd.DataFrame({f"c{col}": [-1.0] * n_rows for col in range(n_cols)})
    schema = pa.DataFrameSchema(
        checks=pa.Check(lambda df: df > 0, n_failure_cases=limit)
    )
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors:
        pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak - before


def test_table_check_failure_limit_allocates_less_than_full_report():
    """A one-row report does not build the full report before truncating it."""
    limited = _table_check_peak_bytes(1)
    full = _table_check_peak_bytes(None)

    # The one-row report skips the per-cell intermediates for every failing
    # cell, so it allocates markedly less than the full report.
    assert limited < full


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_table_check_failure_limit_still_drops_all_invalid_rows(
    n_failure_cases,
):
    """A capped table-level report still removes every failing row."""
    frame = pd.DataFrame(
        {"x": [-1.0, 2.0, -3.0, 4.0], "y": [5.0, -6.0, -7.0, 8.0]},
        index=pd.Index(["a", "b", "c", "d"], name="row"),
    )
    schema = pa.DataFrameSchema(
        checks=pa.Check(lambda df: df > 0, n_failure_cases=n_failure_cases),
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[3]])
    # The input frame is not mutated by the row drop.
    assert list(frame.index) == ["a", "b", "c", "d"]
