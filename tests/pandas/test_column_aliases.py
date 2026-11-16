"""Tests for configurable alternative input headers (``Column(aliases=...)``).

The canonical column name remains the name used by every downstream rule;
``aliases`` only declares extra input headers that are normalized to the
canonical name before the existing validation pipeline runs.
"""

from __future__ import annotations

import copy
import typing

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.errors import (
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
    SchemaInitError,
)
from pandera.typing import Index as PIndex
from pandera.typing import Series

T = typing.TypeVar("T")


@pytest.fixture
def report_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "金额": pa.Column(int, aliases=["费用", "支出"]),
            "备注": pa.Column(str, nullable=True),
        }
    )


class TestColumnAliasInit:
    def test_aliases_stored_as_ordered_tuple(self):
        col = pa.Column(int, aliases=["费用", "支出", "费用"])
        assert col.aliases == ("费用", "支出")

    def test_default_aliases_empty(self):
        assert pa.Column(int).aliases == ()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"regex": True, "name": "col_.*", "aliases": ["x"]},
            {"name": ("a", "b"), "aliases": ["x"]},
        ],
    )
    def test_unsupported_alias_targets_raise(self, kwargs):
        with pytest.raises(SchemaInitError):
            pa.Column(int, **kwargs)

    def test_bare_string_alias_raises(self):
        with pytest.raises(SchemaInitError, match="iterable"):
            pa.Column(int, aliases="费用")

    def test_non_string_alias_raises(self):
        with pytest.raises(SchemaInitError):
            pa.Column(int, aliases=[1])

    def test_alias_equal_to_canonical_name_raises(self):
        with pytest.raises(SchemaInitError):
            pa.Column(int, name="金额", aliases=["金额"])

    def test_empty_aliases_allowed_on_regex_columns(self):
        # properties-driven reconstruction (update_column) keeps working
        col = pa.Column(int, regex=True, name="col_.*")
        assert col.aliases == ()


class TestBasicNormalization:
    def test_alias_header_renamed_to_canonical(self, report_schema):
        df = pd.DataFrame({"费用": [1, 2], "备注": ["a", "b"]})
        out = report_schema.validate(df)
        assert out.columns.tolist() == ["金额", "备注"]
        assert out["金额"].tolist() == [1, 2]

    def test_any_configured_alias_is_accepted(self, report_schema):
        for header in ("费用", "支出"):
            out = report_schema.validate(
                pd.DataFrame({header: [1], "备注": [None]})
            )
            assert out.columns.tolist() == ["金额", "备注"]

    def test_canonical_header_still_accepted(self, report_schema):
        out = report_schema.validate(
            pd.DataFrame({"金额": [1], "备注": [None]})
        )
        assert out.columns.tolist() == ["金额", "备注"]

    def test_non_inplace_preserves_input(self, report_schema):
        df = pd.DataFrame({"费用": [1, 2], "备注": ["a", "b"]})
        report_schema.validate(df)
        assert df.columns.tolist() == ["费用", "备注"]

    def test_non_inplace_preserves_index_and_row_order(self):
        schema = pa.DataFrameSchema({"金额": pa.Column(int, aliases=["费用"])})
        index = pd.Index([30, 10, 20], name="rid")
        df = pd.DataFrame({"费用": [3, 1, 2]}, index=index)
        out = schema.validate(df)
        assert out.index.name == "rid"
        assert out.index.tolist() == [30, 10, 20]
        assert out["金额"].tolist() == [3, 1, 2]

    def test_inplace_does_not_touch_input_on_conflict(self, report_schema):
        df = pd.DataFrame({"金额": [1], "支出": [2]})
        with pytest.raises(SchemaErrors):
            report_schema.validate(df, inplace=True)
        assert df.columns.tolist() == ["金额", "支出"]

    def test_coercion_runs_after_normalization(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(int, coerce=True, aliases=["费用"])}
        )
        out = schema.validate(pd.DataFrame({"费用": ["1", "2"]}))
        assert str(out["金额"].dtype) == "int64"

    def test_checks_run_against_canonical_name(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(int, pa.Check.ge(10), aliases=["费用"])}
        )
        with pytest.raises((SchemaError, SchemaErrors)):
            schema.validate(pd.DataFrame({"费用": [1]}))
        out = schema.validate(pd.DataFrame({"费用": [10]}))
        assert out["金额"].tolist() == [10]


class TestMultipleSources:
    def test_equal_sources_merge_without_canonical(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(int, aliases=["费用", "支出"])}
        )
        df = pd.DataFrame({"费用": [1, 2], "支出": [1, 2]})
        out = schema.validate(df)
        assert out.columns.tolist() == ["金额"]
        assert out["金额"].tolist() == [1, 2]

    def test_equal_sources_merge_with_canonical(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(int, aliases=["费用", "支出"])}
        )
        df = pd.DataFrame({"金额": [1, 2], "费用": [1, 2], "支出": [1, 2]})
        out = schema.validate(df)
        assert out.columns.tolist() == ["金额"]

    def test_equal_missing_values_count_as_equal(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(float, nullable=True, aliases=["费用", "支出"])}
        )
        df = pd.DataFrame({"费用": [1.0, None], "支出": [1.0, None]})
        out = schema.validate(df)
        assert out.columns.tolist() == ["金额"]

    def test_different_missing_positions_conflict(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(float, nullable=True, aliases=["费用", "支出"])}
        )
        df = pd.DataFrame({"费用": [1.0, None], "支出": [None, 1.0]})
        with pytest.raises(SchemaErrors) as exc:
            schema.validate(df)
        assert (
            exc.value.schema_errors[0].reason_code
            is SchemaErrorReason.ALIAS_SOURCE_CONFLICT
        )

    def test_value_conflict_reports_sources(self, report_schema):
        df = pd.DataFrame({"金额": [1, 2], "支出": [3, 2]})
        with pytest.raises(SchemaErrors) as exc:
            report_schema.validate(df)
        error = exc.value.schema_errors[0]
        assert error.reason_code is SchemaErrorReason.ALIAS_SOURCE_CONFLICT
        assert error.failure_cases == ["金额", "支出"]

    def test_dtype_conflict_even_when_values_equal(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(aliases=["费用", "支出"])}
        )
        df = pd.DataFrame(
            {
                "费用": pd.Series([1, 2], dtype="int64"),
                "支出": pd.Series([1, 2], dtype="float64"),
            }
        )
        with pytest.raises(SchemaErrors) as exc:
            schema.validate(df)
        assert (
            exc.value.schema_errors[0].reason_code
            is SchemaErrorReason.ALIAS_SOURCE_CONFLICT
        )

    def test_lazy_collects_all_content_conflicts(self):
        schema = pa.DataFrameSchema(
            {
                "a": pa.Column(aliases=["x", "y"]),
                "b": pa.Column(aliases=["p", "q"]),
            }
        )
        df = pd.DataFrame({"x": [1], "y": [2], "p": [3], "q": [4]})
        with pytest.raises(SchemaErrors) as exc:
            schema.validate(df, lazy=True)
        reasons = [e.reason_code for e in exc.value.schema_errors]
        assert reasons.count(SchemaErrorReason.ALIAS_SOURCE_CONFLICT) == 2


class TestNonUniqueSources:
    def test_duplicate_input_header_is_non_unique(self, report_schema):
        df = pd.DataFrame([[1, 1]], columns=["费用", "费用"])
        with pytest.raises(SchemaErrors) as exc:
            report_schema.validate(df)
        assert (
            exc.value.schema_errors[0].reason_code
            is SchemaErrorReason.ALIAS_SOURCE_NOT_UNIQUE
        )

    def test_two_fields_claim_same_alias(self):
        schema = pa.DataFrameSchema(
            {
                "a": pa.Column(int, aliases=["x"]),
                "b": pa.Column(int, aliases=["x"]),
            }
        )
        with pytest.raises(SchemaErrors) as exc:
            schema.validate(pd.DataFrame({"x": [1]}))
        assert (
            exc.value.schema_errors[0].reason_code
            is SchemaErrorReason.ALIAS_SOURCE_NOT_UNIQUE
        )

    def test_alias_pointing_at_other_canonical_name_conflicts(self):
        schema = pa.DataFrameSchema(
            {
                "a": pa.Column(int, aliases=["b"]),
                "b": pa.Column(int),
            }
        )
        with pytest.raises(SchemaErrors) as exc:
            schema.validate(pd.DataFrame({"a": [1], "b": [2]}))
        assert (
            exc.value.schema_errors[0].reason_code
            is SchemaErrorReason.ALIAS_SOURCE_NOT_UNIQUE
        )

    def test_conflict_error_is_schema_category(self, report_schema):
        df = pd.DataFrame({"金额": [1], "支出": [2]})
        with pytest.raises(SchemaErrors) as exc:
            report_schema.validate(df, lazy=True)
        assert exc.value.error_counts["ALIAS_SOURCE_CONFLICT"] == 1


class TestMissingSources:
    def test_required_column_without_source_still_missing(self, report_schema):
        with pytest.raises(SchemaError) as exc:
            report_schema.validate(pd.DataFrame({"other": [1]}))
        assert (
            exc.value.reason_code is SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
        )

    def test_optional_column_without_source_passes(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(int, required=False, aliases=["费用"])}
        )
        out = schema.validate(pd.DataFrame({"other": [1]}))
        assert "金额" not in out.columns

    def test_add_missing_columns_uses_canonical_name(self):
        schema = pa.DataFrameSchema(
            {
                "金额": pa.Column(
                    int,
                    aliases=["费用"],
                    default=0,
                    nullable=True,
                )
            },
            add_missing_columns=True,
        )
        out = schema.validate(pd.DataFrame({"other": [1, 2]}))
        assert out["金额"].tolist() == [0, 0]


class TestStrictAndOrder:
    def test_consumed_alias_not_an_extra_column_in_strict_mode(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(int, aliases=["费用", "支出"])},
            strict=True,
        )
        df = pd.DataFrame({"费用": [1], "支出": [1], "junk": [2]})
        with pytest.raises(SchemaErrors) as exc:
            schema.validate(df)
        # only the truly unknown header is reported
        cases = [
            e.failure_cases
            for e in exc.value.schema_errors
            if e.reason_code is SchemaErrorReason.COLUMN_NOT_IN_SCHEMA
        ]
        assert cases == ["junk"]

    def test_strict_filter_drops_only_unknown_columns(self):
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(int, aliases=["费用"])},
            strict="filter",
        )
        out = schema.validate(pd.DataFrame({"费用": [1], "junk": [9]}))
        assert out.columns.tolist() == ["金额"]

    def test_ordered_uses_canonical_names(self):
        schema = pa.DataFrameSchema(
            {
                "a": pa.Column(int, aliases=["x"]),
                "b": pa.Column(int),
            },
            ordered=True,
        )
        out = schema.validate(pd.DataFrame({"x": [1], "b": [2]}))
        assert out.columns.tolist() == ["a", "b"]

    def test_ordered_violation_after_normalization_reported(self):
        schema = pa.DataFrameSchema(
            {
                "a": pa.Column(int, aliases=["x"]),
                "b": pa.Column(int),
            },
            ordered=True,
        )
        with pytest.raises(SchemaErrors) as exc:
            schema.validate(pd.DataFrame({"b": [2], "x": [1]}))
        reasons = [e.reason_code for e in exc.value.schema_errors]
        assert SchemaErrorReason.COLUMN_NOT_ORDERED in reasons


class TestDataFrameModelAliases:
    def test_field_aliases_build_column_aliases(self):
        class Report(pa.DataFrameModel):
            amount: int = pa.Field(alias="金额", aliases=["费用", "支出"])

        schema = Report.to_schema()
        assert list(schema.columns) == ["金额"]
        assert schema.columns["金额"].aliases == ("费用", "支出")
        out = Report.validate(pd.DataFrame({"费用": [1, 2]}))
        assert out.columns.tolist() == ["金额"]

    def test_field_alias_remains_canonical_name(self):
        class Report(pa.DataFrameModel):
            amount: int = pa.Field(alias="金额", aliases=["费用"])

        assert Report.to_schema().columns["金额"].name == "金额"

    def test_index_field_aliases_rejected(self):
        class Bad(pa.DataFrameModel):
            idx: PIndex[int] = pa.Field(aliases=["x"])

        with pytest.raises(SchemaInitError):
            Bad.to_schema()

    def test_model_generated_schema_roundtrip_yaml(self):
        class Report(pa.DataFrameModel):
            金额: int = pa.Field(aliases=["费用"])

        schema = Report.to_schema()
        restored = pa.DataFrameSchema.from_yaml(schema.to_yaml())
        assert restored == schema

    def test_aliases_survive_model_inheritance(self):
        class Base(pa.DataFrameModel):
            a: int = pa.Field(aliases=["x"])

        class Child(Base):
            b: str

        schema = Child.to_schema()
        assert schema.columns["a"].aliases == ("x",)
        out = Child.validate(pd.DataFrame({"x": [1], "b": ["z"]}))
        assert out.columns.tolist() == ["a", "b"]

    def test_aliases_survive_generic_parameterization(self):
        class GenericModel(pa.DataFrameModel, typing.Generic[T]):
            col: T = pa.Field(aliases=["legacy"])

        parameterized = GenericModel[int]
        schema = parameterized.to_schema()
        assert schema.columns["col"].aliases == ("legacy",)
        out = parameterized.validate(pd.DataFrame({"legacy": [1, 2]}))
        assert out.columns.tolist() == ["col"]


class TestSerialization:
    @pytest.fixture
    def aliased_schema(self):
        return pa.DataFrameSchema(
            {
                "金额": pa.Column(int, coerce=True, aliases=["费用", "支出"]),
                "note": pa.Column(str, nullable=True),
            },
            strict=True,
        )

    @pytest.mark.parametrize("minimal", [True, False])
    def test_yaml_roundtrip(self, aliased_schema, minimal):
        payload = aliased_schema.to_yaml(minimal=minimal)
        restored = pa.DataFrameSchema.from_yaml(payload)
        assert restored == aliased_schema

    @pytest.mark.parametrize("minimal", [True, False])
    def test_json_roundtrip(self, aliased_schema, minimal):
        payload = aliased_schema.to_json(minimal=minimal)
        restored = pa.DataFrameSchema.from_json(payload)
        assert restored == aliased_schema

    def test_minimal_output_omits_empty_aliases(self):
        schema = pa.DataFrameSchema({"a": pa.Column(int)})
        payload = schema.to_yaml()
        assert "aliases" not in payload

    def test_legacy_schema_without_aliases_unchanged(self):
        schema = pa.DataFrameSchema({"a": pa.Column(int)})
        assert pa.DataFrameSchema.from_yaml(schema.to_yaml()) == schema

    def test_minimal_schema_script_contains_aliases(self, aliased_schema):
        from pandera.io import common_io

        script = common_io._to_script_minimal(aliased_schema, backend="pandas")
        assert "aliases=" in script
        namespace = {}
        exec(script, namespace)
        assert namespace["schema"] == aliased_schema

    def test_model_script_contains_aliases(self, aliased_schema):
        from pandera.io._script_model import (
            to_dataframe_model_script,
        )

        script = to_dataframe_model_script(
            aliased_schema, "pandas", minimal=True
        )
        assert "aliases=" in script
        # annotations are resolved against real module globals, where the
        # top-level ``Series`` import above is visible.
        local_namespace: dict = {}
        exec(script, globals(), local_namespace)
        rebuilt = local_namespace["GeneratedModel"].to_schema()
        assert rebuilt.columns["金额"].aliases == ("费用", "支出")


class TestSchemaTransforms:
    @pytest.fixture
    def schema(self):
        return pa.DataFrameSchema(
            {"金额": pa.Column(int, aliases=["费用", "支出"])}
        )

    def test_deepcopy_preserves_aliases(self, schema):
        assert copy.deepcopy(schema).columns["金额"].aliases == (
            "费用",
            "支出",
        )

    def test_add_columns_preserves_aliases(self, schema):
        expanded = schema.add_columns({"x": pa.Column(str)})
        assert expanded.columns["金额"].aliases == ("费用", "支出")

    def test_update_column_preserves_aliases(self, schema):
        updated = schema.update_column("金额", nullable=True)
        assert updated.columns["金额"].aliases == ("费用", "支出")
        assert updated.columns["金额"].nullable is True

    def test_rename_columns_preserves_aliases(self, schema):
        renamed = schema.rename_columns({"金额": "amount"})
        assert renamed.columns["amount"].aliases == ("费用", "支出")

    def test_select_columns_preserves_aliases(self, schema):
        selected = schema.select_columns(["金额"])
        assert selected.columns["金额"].aliases == ("费用", "支出")


class TestParsersAndParsedColumns:
    def test_schema_parser_references_canonical_name(self):
        parser = pa.Parser(
            lambda df: df.assign(金额=df["金额"] * 2), target="金额"
        )
        schema = pa.DataFrameSchema(
            {"金额": pa.Column(int, aliases=["费用"])},
            parsers=[parser],
        )
        out = schema.validate(pd.DataFrame({"费用": [1, 2, 3]}))
        assert out["金额"].tolist() == [2, 4, 6]

    def test_parsed_field_source_uses_canonical_name(self):
        class Report(pa.DataFrameModel):
            金额: int = pa.Field(aliases=["费用"])
            doubled: int = pa.ParsedField(
                parser=lambda s: s * 2, source="金额"
            )

        out = Report.validate(pd.DataFrame({"费用": [1, 2]}))
        assert out["doubled"].tolist() == [2, 4]


class TestReportRuleReuse:
    def test_same_rules_cover_legacy_and_canonical_headers(self):
        rule = pa.DataFrameSchema(
            {
                "金额": pa.Column(
                    float, nullable=True, aliases=["费用", "支出"]
                ),
                "部门": pa.Column(str),
            },
            checks=[pa.Check(lambda df: df["金额"].notna().all())],
            strict=True,
        )
        old_report = pd.DataFrame({"费用": [100.0, 200.0], "部门": ["a", "b"]})
        new_report = pd.DataFrame({"金额": [100.0, 200.0], "部门": ["a", "b"]})
        validated_old = rule.validate(old_report)
        validated_new = rule.validate(new_report)
        assert validated_old.columns.tolist() == ["金额", "部门"]
        assert validated_old.equals(validated_new)
        # legacy input itself is not mutated
        assert old_report.columns.tolist() == ["费用", "部门"]


class TestNoAliasesLegacyBehavior:
    def test_schema_without_aliases_validates_as_before(self):
        schema = pa.DataFrameSchema({"a": pa.Column(int), "b": pa.Column(str)})
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        out = schema.validate(df)
        assert out.columns.tolist() == ["a", "b"]
        # input untouched even though normalization had nothing to do
        assert df.columns.tolist() == ["a", "b"]

    def test_regex_columns_unaffected(self):
        schema = pa.DataFrameSchema({"col_.*": pa.Column(int, regex=True)})
        df = pd.DataFrame({"col_1": [1], "col_2": [2]})
        out = schema.validate(df)
        assert out.columns.tolist() == ["col_1", "col_2"]
