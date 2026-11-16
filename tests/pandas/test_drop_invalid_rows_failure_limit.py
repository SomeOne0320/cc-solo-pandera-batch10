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


# ---------------------------------------------------------------------------
# Duplicate index labels: rows are removed by position, not by label.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_failure_cases", [0, 1, 2, None])
def test_drop_invalid_rows_duplicate_index_removes_only_failing_rows(
    n_failure_cases,
):
    """A repeated label does not make a passing row collateral damage."""
    frame = pd.DataFrame(
        {"x": [1, -2, 3, -4]}, index=pd.Index([0, 0, 1, 1], name="row")
    )
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[0, 2]])
    assert validated.index.names == ["row"]
    assert validated.index.equals(pd.Index([0, 1], name="row"))


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_duplicate_index_multiple_checks_union(
    n_failure_cases,
):
    """Overlapping failures across checks are combined before removal."""
    frame = pd.DataFrame(
        {"x": [-1, 2, -3, 4], "y": [1, -2, -3, 4]},
        index=pd.Index(["a", "a", "b", "b"]),
    )
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int, pa.Check.gt(0, n_failure_cases=n_failure_cases)
            ),
            "y": pa.Column(
                int, pa.Check.gt(0, n_failure_cases=n_failure_cases)
            ),
        },
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[3]])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_duplicate_multiindex_combinations(n_failure_cases):
    """Repeated MultiIndex tuples are treated like any other repeated label."""
    index = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 1), ("b", 2), ("b", 2)], names=["group", "row"]
    )
    frame = pd.DataFrame({"x": [1, -2, 3, -4]}, index=index)
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[0, 2]])
    assert validated.index.names == ["group", "row"]


@pytest.mark.parametrize(
    "schema, expected_selector",
    [
        # dtype violations
        (
            pa.DataFrameSchema({"x": pa.Column(int)}),
            [0, 2],
        ),
        # nullability violations
        (
            pa.DataFrameSchema({"x": pa.Column(float, nullable=False)}),
            [0, 2],
        ),
    ],
)
def test_drop_invalid_rows_duplicate_index_core_checks(
    schema, expected_selector
):
    """Core-check failures also map back to exact rows, not labels."""
    frame = pd.DataFrame(
        {"x": [1, "bad", 2, "bad"]}, index=pd.Index([0, 0, 1, 1])
    )
    schema = schema.update_column("x", dtype=schema.columns["x"].dtype)
    schema.drop_invalid_rows = True

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[expected_selector])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_duplicate_index_ignore_na(n_failure_cases):
    """Null-skipping rows keep their position among repeated labels."""
    frame = pd.DataFrame(
        {"x": [-1.0, None, -2.0, 3.0]}, index=pd.Index([0, 0, 1, 1])
    )
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                float,
                pa.Check(
                    lambda obj: obj > 0,
                    ignore_na=True,
                    n_failure_cases=n_failure_cases,
                ),
                nullable=True,
            )
        },
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[[1, 3]])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_duplicate_index_head_subsample(n_failure_cases):
    """Only checked rows are dropped; same-label unchecked rows are kept."""
    frame = pd.DataFrame({"x": [-1, -2, -3, 4]}, index=pd.Index([0, 0, 0, 0]))
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=n_failure_cases))},
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, head=2, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[2:])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_duplicate_index_nested_column_subsample(
    n_failure_cases,
):
    """A nested column rule drops only the checked subset."""
    frame = pd.DataFrame({"x": [-1, -2, -3, 4]}, index=pd.Index([0, 0, 0, 0]))
    schema = pa.DataFrameSchema(
        {
            "x": pa.Column(
                int,
                pa.Check.gt(0, n_failure_cases=n_failure_cases),
                drop_invalid_rows=True,
            )
        }
    )

    validated = schema.validate(frame, head=2, lazy=True)

    pd.testing.assert_frame_equal(validated, frame.iloc[2:])


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
@pytest.mark.parametrize("schema_type", ["series", "column", "dataframe"])
def test_drop_invalid_rows_duplicate_index_schema_types(
    n_failure_cases, schema_type
):
    """Series, standalone Column, and DataFrame schemas all drop by row."""
    frame = pd.DataFrame({"x": [1, -2, 3, -4]}, index=pd.Index([0, 0, 1, 1]))
    check = pa.Check.gt(0, n_failure_cases=n_failure_cases)
    if schema_type == "series":
        schema = pa.SeriesSchema(int, check, drop_invalid_rows=True)
        data = frame["x"]
    elif schema_type == "column":
        schema = pa.Column(int, check, name="x", drop_invalid_rows=True)
        data = frame
    else:
        schema = pa.DataFrameSchema(
            {"x": pa.Column(int, check)}, drop_invalid_rows=True
        )
        data = frame

    validated = schema.validate(data, lazy=True)

    expected = (
        frame["x"].iloc[[0, 2]]
        if schema_type == "series"
        else frame.iloc[[0, 2]]
    )
    if schema_type == "series":
        pd.testing.assert_series_equal(validated, expected)
    else:
        pd.testing.assert_frame_equal(validated, expected)


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_duplicate_index_parser(n_failure_cases):
    """Parsed values survive and duplicate labels stay distinct."""
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "channel": ["Google", "bing", "invalid", "google"],
        },
        index=pd.Index([0, 0, 1, 1]),
    )
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(int),
            "channel": pa.Column(
                str,
                pa.Check.isin(
                    ["google", "bing"], n_failure_cases=n_failure_cases
                ),
                parsers=pa.Parser(lambda s: s.str.lower()),
            ),
        },
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    expected = frame.iloc[[0, 1, 3]].copy()
    expected["channel"] = ["google", "bing", "google"]
    pd.testing.assert_frame_equal(validated, expected)


def test_drop_invalid_rows_duplicate_index_does_not_mutate_input():
    """The default non-inplace call leaves the caller's frame untouched."""
    frame = pd.DataFrame({"x": [1, -2, 3, -4]}, index=pd.Index([0, 0, 1, 1]))
    original = frame.copy(deep=True)
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0))}, drop_invalid_rows=True
    )

    schema.validate(frame, lazy=True)

    pd.testing.assert_frame_equal(frame, original)


def test_drop_invalid_rows_duplicate_index_failure_report_labels():
    """Reports keep the user-visible labels, never internal row positions."""
    frame = pd.DataFrame({"x": [-1, -2, -3]}, index=pd.Index(["a", "a", "b"]))
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0, n_failure_cases=1))}
    )

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        schema.validate(frame, lazy=True)

    failure_cases = exc_info.value.failure_cases
    assert len(failure_cases) == 1
    assert list(failure_cases["index"]) == ["a"]


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_duplicate_index_coercion(n_failure_cases):
    """Coercion failures drop only the rows that could not be coerced."""
    frame = pd.DataFrame(
        {"x": ["1", "bad", "2", "bad"]}, index=pd.Index([0, 0, 1, 1])
    )
    schema = pa.DataFrameSchema(
        {"x": pa.Column(int, coerce=True)}, drop_invalid_rows=True
    )

    validated = schema.validate(frame, lazy=True)

    # A failed coercion leaves the column uncoerced, so the surviving rows
    # keep their original values and dtype.
    expected = frame.iloc[[0, 2]]
    pd.testing.assert_frame_equal(validated, expected)


@pytest.mark.parametrize("n_failure_cases", [0, 1, None])
def test_drop_invalid_rows_duplicate_index_dataframe_dtype_coercion(
    n_failure_cases,
):
    """Frame-level coercion masks are reduced per row, not per cell."""
    frame = pd.DataFrame(
        {
            "a": ["1", "bad", "2", "bad"],
            "b": ["3", "4", "5", "bad"],
        },
        index=pd.Index([0, 0, 1, 1]),
    )
    schema = pa.DataFrameSchema(
        {"a": pa.Column(), "b": pa.Column()},
        dtype=int,
        coerce=True,
        drop_invalid_rows=True,
    )

    validated = schema.validate(frame, lazy=True)

    expected = frame.iloc[[0, 2]]
    pd.testing.assert_frame_equal(validated, expected)


def test_drop_invalid_rows_duplicate_index_no_internal_attrs_leaked():
    """Internal row bookkeeping never shows up in output or error objects."""
    frame = pd.DataFrame({"x": [-1, -2]}, index=pd.Index([0, 0]))
    throwing = pa.DataFrameSchema({"x": pa.Column(int, pa.Check.gt(0))})
    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        throwing.validate(frame, lazy=True)
    assert exc_info.value.failure_cases.attrs == {}
    assert exc_info.value.data.attrs == {}

    dropping = pa.DataFrameSchema(
        {"x": pa.Column(int, pa.Check.gt(0))}, drop_invalid_rows=True
    )
    validated = dropping.validate(frame, lazy=True)
    assert validated.attrs == {}
