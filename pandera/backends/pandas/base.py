"""Pandas parsing, validation, and error-reporting backends."""

import ast
import functools
import warnings
from collections import defaultdict
from contextvars import ContextVar
from typing import Any, Optional, TypeVar, Union

import numpy as np
import pandas as pd

# Context var to preserve parsed column values when drop_invalid_rows is used
# with custom parsers (fixes #2216: parsed values otherwise lost after drop).
_parsed_column_values: ContextVar[dict[str, pd.Series] | None] = ContextVar(
    "pandera_parsed_column_values", default=None
)

# Internal attrs keys used to distinguish rows that share an index label.
# ``_ROW_IDS_ATTR`` stores positional row ids on a validated object, and
# ``_FAILING_IDS_ATTR`` stores the ids of exactly the rows that failed a
# check. They are internal only: never rendered in failure reports and
# removed from the object returned by the outermost ``validate`` call.
_ROW_IDS_ATTR = "_pandera_row_ids"
_FAILING_IDS_ATTR = "_pandera_failing_row_ids"

# Nesting depth of backend ``validate`` calls. Row-id bookkeeping is stripped
# only when the outermost validation returns.
_validation_depth: ContextVar[int] = ContextVar(
    "pandera_row_ids_validation_depth", default=0
)


def strip_row_ids_on_exit(validate_fn):
    """Strip internal row-id bookkeeping from an outermost validate result."""

    @functools.wraps(validate_fn)
    def wrapper(self, check_obj, *args, **kwargs):
        depth = _validation_depth.get()
        _validation_depth.set(depth + 1)
        try:
            try:
                result = validate_fn(self, check_obj, *args, **kwargs)
            except SchemaErrors as exc:
                if depth == 0:
                    _strip_schema_errors_row_ids(exc)
                raise
            except SchemaError as exc:
                if depth == 0:
                    _strip_schema_error_row_ids(exc)
                raise
        finally:
            _validation_depth.set(depth)
        if depth != 0:
            return result
        # ``inplace=True`` may leave the bookkeeping on the caller's object.
        if kwargs.get("inplace"):
            strip_row_ids(check_obj)
        return strip_row_ids(result)

    return wrapper


def _strip_schema_errors_row_ids(exc):
    """Remove internal row-id attrs from a raised ``SchemaErrors``."""
    for schema_error in getattr(exc, "schema_errors", ()):
        _strip_schema_error_row_ids(schema_error)
    data = getattr(exc, "data", None)
    if data is not None:
        exc.data = strip_row_ids(data)
    # The consolidated failure cases are rebuilt from the individual errors,
    # so sanitize both the attribute and the message view.
    if getattr(exc, "failure_cases", None) is not None:
        exc.failure_cases = strip_row_ids(exc.failure_cases)


def _strip_schema_error_row_ids(exc):
    """Remove internal row-id attrs from a single ``SchemaError``."""
    exc.failure_cases = strip_row_ids(exc.failure_cases)
    data = getattr(exc, "data", None)
    if data is not None:
        exc.data = strip_row_ids(data)


def _supports_row_ids(check_obj) -> bool:
    """Whether positional row ids are safe to use for ``check_obj``.

    Only native pandas objects propagate ``attrs`` alongside their rows;
    other pandas-like backends keep the legacy label-based behavior.
    """
    if not isinstance(check_obj, (pd.DataFrame, pd.Series)):
        return False
    module = type(check_obj).__module__
    return module.split(".")[0] in ("pandas", "geopandas")


def _set_row_ids(check_obj, row_ids):
    """Store positional row ids on ``check_obj`` via ``.attrs``.

    Ids are positions in the frame being validated, so they tell apart rows
    that share an index label. They are stored as a tuple so pandas' own
    ``attrs`` comparison (e.g. in ``pd.concat``) stays well-defined.
    """
    try:
        check_obj.attrs[_ROW_IDS_ATTR] = tuple(int(i) for i in row_ids)
    except Exception:  # pragma: no cover - non-pandas objects
        pass
    return check_obj


def _read_row_ids(check_obj) -> np.ndarray | None:
    try:
        row_ids = check_obj.attrs.get(_ROW_IDS_ATTR)
    except Exception:  # pragma: no cover - non-pandas objects
        return None
    if not isinstance(row_ids, tuple):
        return None
    return np.asarray(row_ids, dtype=np.int64)


def get_row_ids(check_obj) -> np.ndarray | None:
    """Return the positional row ids attached to ``check_obj``, if any."""
    row_ids = _read_row_ids(check_obj)
    if row_ids is None or len(row_ids) != len(check_obj):
        return None
    return row_ids


def ensure_row_ids(check_obj):
    """Tag ``check_obj`` with positional row ids unless it already has them.

    Objects that already carry ids keep them, which keeps rows dropped while
    checking a subsample attributable to positions in the validated object.
    """
    if not _supports_row_ids(check_obj):
        return check_obj
    if get_row_ids(check_obj) is None:
        _set_row_ids(check_obj, np.arange(len(check_obj), dtype=np.int64))
    return check_obj


def take_rows(check_obj, mask):
    """Select rows positionally, preserving the index and row order.

    The kept rows retain their original ids so that subsequent nested
    validations keep referring to the same rows.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    positions = np.flatnonzero(mask)
    row_ids = _read_row_ids(check_obj)
    result = check_obj.iloc[positions]
    if row_ids is not None:
        result = result.copy()
        _set_row_ids(result, row_ids[positions])
    return result


def strip_row_ids(check_obj):
    """Remove internal row-id bookkeeping from a returned check object."""
    try:
        attrs = check_obj.attrs
    except Exception:  # pragma: no cover - non-pandas objects
        return check_obj
    if _ROW_IDS_ATTR not in attrs and _FAILING_IDS_ATTR not in attrs:
        return check_obj
    try:
        check_obj.attrs = {
            k: v
            for k, v in attrs.items()
            if k not in (_ROW_IDS_ATTR, _FAILING_IDS_ATTR)
        }
    except Exception:  # pragma: no cover
        pass
    return check_obj


def get_failing_ids(failure_cases) -> np.ndarray | None:
    """Read the row ids recorded for a check's failing rows.

    The recorded ids are intentionally independent of the reported row count:
    a report truncated by ``n_failure_cases`` still describes the same failing
    rows.
    """
    try:
        ids = failure_cases.attrs.get(_FAILING_IDS_ATTR)
    except Exception:  # pragma: no cover - non-pandas objects
        return None
    if not isinstance(ids, tuple):
        return None
    return np.asarray(ids, dtype=np.int64)


def _has_failing_ids(failure_cases) -> bool:
    """Whether failure bookkeeping carries the exact failing row ids."""
    try:
        return _FAILING_IDS_ATTR in failure_cases.attrs
    except Exception:  # pragma: no cover - non-pandas objects
        return False


def record_failing_ids(failure_cases, ids):
    """Record which rows failed a check on a copy of its failure cases.

    The ids are deliberately independent of the reported row count: a
    truncated report still describes the same failing rows.
    """
    if failure_cases is None or ids is None:
        return failure_cases
    if not isinstance(failure_cases, (pd.Series, pd.DataFrame)):
        return failure_cases
    try:
        failure_cases = failure_cases.copy()
        failure_cases.attrs[_FAILING_IDS_ATTR] = tuple(
            int(i) for i in np.asarray(ids, dtype=np.int64)
        )
    except Exception:  # pragma: no cover - non-pandas objects
        return failure_cases
    return failure_cases


def output_failure_mask(check_output) -> np.ndarray | None:
    """Boolean mask of failing positions from a check output, if possible."""
    if not isinstance(check_output, (pd.Series, pd.DataFrame)):
        return None
    if isinstance(check_output, pd.DataFrame):
        try:
            check_output = check_output.all(axis="columns")
        except (TypeError, ValueError):  # pragma: no cover - non-boolean
            return None
    try:
        return ~np.asarray(check_output).astype(bool)
    except (TypeError, ValueError):  # pragma: no cover - non-boolean output
        return None


def ids_from_mask(check_obj, mask) -> np.ndarray | None:
    """Map a positional failure mask onto ``check_obj``'s row ids."""
    if mask is None:
        return None
    row_ids = get_row_ids(check_obj)
    if row_ids is None:
        return None
    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.ndim > 1:
        # A per-cell mask: a row fails if any of its cells fails.
        mask = mask.any(axis=tuple(range(1, mask.ndim)))
    if len(mask) != len(check_obj):
        return None
    return row_ids[mask]


def _hashable(label):
    """Normalize an index label (or rendered label) into a dict key."""
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, list):
        return tuple(_hashable(item) for item in label)
    if isinstance(label, tuple):
        return tuple(_hashable(item) for item in label)
    return label


def _index_key(label):
    """Key for an actual index label (tuples for a MultiIndex)."""
    if isinstance(label, tuple):
        return tuple(_hashable(item) for item in label)
    return _hashable(label)


def _reported_label_keys(label):
    """Candidate keys for a label rendered into a failure report.

    ``reshape_failure_cases`` stringifies MultiIndex labels, so try both the
    literal-evaluated tuple and the raw rendered value.
    """
    keys = []
    if isinstance(label, str):
        stripped = label.strip()
        if stripped.startswith(("(", "[")) and stripped.endswith((")", "]")):
            try:
                keys.append(_hashable(ast.literal_eval(stripped)))
            except (ValueError, SyntaxError):
                pass
        keys.append(_hashable(label))
    else:
        keys.append(_hashable(label))
    return keys


def labels_mask(check_obj, labels) -> np.ndarray | None:
    """Boolean mask of rows whose index label is reported as failing.

    Labels are consumed one occurrence at a time, so repeated labels map to
    distinct rows instead of all resolving to the first occurrence.
    """
    if labels is None:
        return None
    index_keys = [_index_key(label) for label in check_obj.index]

    positions_by_label: dict[Any, list[int]] = defaultdict(list)
    for pos, key in enumerate(index_keys):
        positions_by_label[key].append(pos)

    seen: dict[Any, int] = defaultdict(int)
    mask = np.zeros(len(index_keys), dtype=bool)
    for label in labels:
        for key in _reported_label_keys(label):
            candidates = positions_by_label.get(key)
            if not candidates:
                continue
            seen[key] += 1
            occurrences = len(candidates)
            mask[candidates[min(seen[key] - 1, occurrences - 1)]] = True
            break
    return mask


def ids_from_labels(check_obj, labels) -> np.ndarray | None:
    """Map reported index labels onto ``check_obj``'s row ids."""
    mask = labels_mask(check_obj, labels)
    if mask is None:
        return None
    row_ids = get_row_ids(check_obj)
    if row_ids is None:
        row_ids = np.arange(len(check_obj), dtype=np.int64)
        if len(row_ids) != len(mask):
            return None
    return row_ids[mask]


def failure_labels(failure_cases) -> list | None:
    """Extract the reported row labels from reshaped failure cases."""
    if isinstance(failure_cases, pd.Series):
        return list(failure_cases.index)
    if isinstance(failure_cases, pd.DataFrame):
        if "index" in failure_cases.columns:
            return list(failure_cases["index"])
    return None


def attach_failing_ids(
    check_obj, failure_cases, check_output=None, failing_ids=None
) -> Any:
    """Record which rows of ``check_obj`` failed a check.

    The ids computed by the check backend cover every failing row even when
    the report is truncated by ``n_failure_cases``. Failing that, the full
    check output is resolved against ``check_obj``, and as a last resort the
    reported labels are matched. Reported label matching consumes labels
    occurrence by occurrence, so repeated labels map to distinct rows and
    only the rows actually checked are ever affected.
    """
    if get_failing_ids(failure_cases) is not None:
        return failure_cases
    if not isinstance(failure_cases, (pd.Series, pd.DataFrame)):
        return failure_cases

    ids = None
    if failing_ids is not None:
        ids = np.asarray(failing_ids, dtype=np.int64)
    if ids is None:
        ids = ids_from_mask(check_obj, output_failure_mask(check_output))
    if ids is None:
        ids = ids_from_labels(check_obj, failure_labels(failure_cases))
    return record_failing_ids(failure_cases, ids)


from pandera.api.base.checks import CheckResult
from pandera.api.base.error_handler import ErrorHandler
from pandera.api.parsers import Parser
from pandera.backends.base import (
    BaseSchemaBackend,
    CoreCheckResult,
    CoreParserResult,
)
from pandera.backends.pandas.error_formatters import (
    consolidate_failure_cases,
    format_generic_error_message,
    format_vectorized_error_message,
    reshape_failure_cases,
)
from pandera.errors import (
    FailureCaseMetadata,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
    SchemaWarning,
)

FieldCheckObj = Union[pd.Series, pd.DataFrame]

T = TypeVar(
    "T",
    pd.Series,
    pd.DataFrame,
    FieldCheckObj,
    covariant=True,
)


class PandasSchemaBackend(BaseSchemaBackend):
    """Base backend for pandas schemas."""

    def subsample(
        self,
        check_obj,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
    ):
        if head is None and tail is None and sample is None:
            return check_obj

        row_ids = (
            get_row_ids(check_obj) if _supports_row_ids(check_obj) else None
        )
        if row_ids is None:
            pandas_obj_subsample = []
            if head is not None:
                pandas_obj_subsample.append(check_obj.head(head))
            if tail is not None:
                pandas_obj_subsample.append(check_obj.tail(tail))
            if sample is not None:
                pandas_obj_subsample.append(
                    check_obj.sample(sample, random_state=random_state)
                )
            return pd.concat(pandas_obj_subsample).pipe(
                lambda x: x[~x.index.duplicated()]
            )

        # Select by position so that repeated index labels neither collapse
        # distinct sampled rows nor evict rows that were truly selected.
        parts = []
        selected_ids = []
        n_rows = len(check_obj)
        if head is not None:
            part = check_obj.head(head)
            parts.append(part)
            selected_ids.append(row_ids[: len(part)])
        if tail is not None:
            part = check_obj.tail(tail)
            parts.append(part)
            selected_ids.append(row_ids[n_rows - len(part) :])
        if sample is not None:
            positions = (
                pd.Series(np.arange(n_rows))
                .sample(sample, random_state=random_state)
                .to_numpy()
            )
            parts.append(check_obj.iloc[positions])
            selected_ids.append(row_ids[positions])

        combined_ids = np.concatenate(selected_ids)
        keep = np.flatnonzero(~pd.Series(combined_ids).duplicated())
        result = pd.concat(parts).iloc[keep]
        _set_row_ids(result, combined_ids[keep])
        return result

    def run_parser(
        self,
        check_obj,
        parser: Parser,
        parser_index: int,
        *args,
    ) -> CoreParserResult:
        """Handle parser results.

        :param check_obj: data object to be validated.
        :param schema: pandera schema object
        :param parser: Parser object used to validate pandas object.
        :param parser_index: index of parser in the schema component parser list.
        :param args: arguments to pass into parser object.
        :returns:  ParserResult
        """
        parser_result = parser(check_obj, *args)

        return CoreParserResult(
            passed=True,
            parser=parser,
            parser_index=parser_index,
            parser_output=parser_result.parser_output,
            reason_code=SchemaErrorReason.DATAFRAME_PARSER,
            failure_cases=None,
            message=None,
        )

    def run_check(
        self,
        check_obj,
        schema,
        check,
        check_index: int,
        *args,
    ) -> CoreCheckResult:
        """Handle check results, raising SchemaError on check failure.

        :param check_obj: data object to be validated.
        :param schema: pandera schema object.
        :param check: Check object used to validate pandas object.
        :param check_index: index of check in the schema component check list.
        :param args: arguments to pass into check object.
        :returns: True if check results pass or check.raise_warning=True, otherwise
            False.
        """
        check_result: CheckResult = check(check_obj, *args)

        passed = check_result.check_passed
        failure_cases = None
        message = None

        if not passed:
            if check_result.failure_cases is None:
                failure_cases = check_result.check_passed
                message = format_generic_error_message(
                    schema, check, check_index
                )
            else:
                failure_cases = reshape_failure_cases(
                    check_result.failure_cases, check.ignore_na
                )
                message = format_vectorized_error_message(
                    schema, check, check_index, failure_cases
                )
                # Record the ids of every failing row before the report can
                # be truncated, so removal is not limited by n_failure_cases.
                failure_cases = attach_failing_ids(
                    check_obj,
                    failure_cases,
                    check_result.check_output,
                    check_result.failing_ids,
                )

            # raise a warning without exiting if the check is specified to do so
            # but make sure the check passes
            if check.raise_warning:
                warnings.warn(
                    message,
                    SchemaWarning,
                    stacklevel=2,
                )
                return CoreCheckResult(
                    passed=True,
                    check=check,
                    reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                )

        return CoreCheckResult(
            passed=passed,
            check=check,
            check_index=check_index,
            check_output=check_result.check_output,
            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
            message=message,
            failure_cases=failure_cases,
        )

    def failure_cases_metadata(
        self,
        schema_name: str,
        schema_errors: list[SchemaError],
    ) -> FailureCaseMetadata:
        """Create failure cases metadata required for SchemaErrors exception."""
        failure_cases = consolidate_failure_cases(schema_errors)

        error_handler = ErrorHandler()
        error_handler.collect_errors(schema_errors)
        error_dicts = {}

        def defaultdict_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: defaultdict_to_dict(v) for k, v in d.items()}
            return d

        if error_handler.collected_errors:
            error_dicts = error_handler.summarize(schema_name=schema_name)
            error_dicts = defaultdict_to_dict(error_dicts)

        error_counts = defaultdict(int)  # type: ignore
        for error in error_handler.collected_errors:
            error_counts[error["reason_code"].name] += 1

        return FailureCaseMetadata(
            failure_cases=failure_cases,
            message=error_dicts,
            error_counts=error_counts,
        )

    def drop_invalid_rows(self, check_obj, error_handler: ErrorHandler):
        """Remove invalid elements in a check obj according to failures caught by the error handler.

        Rows are removed by position, not by index label: ``Index.isin`` would
        also discard unrelated rows that merely share a failing label. All
        failing rows are collected first and removed in one step, so the union
        of failures across checks is dropped regardless of how many failure
        cases each report shows.
        """
        if not (
            _supports_row_ids(check_obj)
            or any(
                _has_failing_ids(err.failure_cases)
                for err in error_handler.schema_errors
            )
        ):
            # Pandas-like backends beyond native pandas don't propagate attrs,
            # so keep their established label-based behavior.
            check_obj = self._drop_invalid_rows_by_label(
                check_obj, error_handler
            )
        else:
            failing_ids = self._collect_failing_ids(check_obj, error_handler)
            if failing_ids is not None:
                row_ids = get_row_ids(check_obj)
                position_ids = (
                    row_ids
                    if row_ids is not None
                    else np.arange(len(check_obj), dtype=np.int64)
                )
                failing = set(np.asarray(failing_ids, dtype=np.int64).tolist())
                mask = np.array(
                    [row_id not in failing for row_id in position_ids],
                    dtype=bool,
                )
                check_obj = take_rows(check_obj, mask)

        # Restore parsed column values that may have been overwritten (e.g. custom
        # parser + drop_invalid_rows returning None in parsed column, #2216).
        parsed = _parsed_column_values.get()
        if parsed:
            for colname, series in parsed.items():
                if colname not in check_obj.columns:
                    continue
                try:
                    restored = series.reindex(check_obj.index)
                    check_obj.loc[:, colname] = restored.values
                except Exception:
                    pass
            _parsed_column_values.set(None)

        return check_obj

    @staticmethod
    def _drop_invalid_rows_by_label(check_obj, error_handler):
        """Legacy label-based removal for non-native pandas objects."""
        for err in error_handler.schema_errors:
            if isinstance(err.failure_cases, str):
                # if the failure cases are a string, it means the error is
                # a schema-level error.
                continue
            # Reporting can truncate or omit failure cases from the full mask.
            if isinstance(
                err.failure_cases, (pd.Series, pd.DataFrame)
            ) and isinstance(err.check_output, (pd.Series, pd.DataFrame)):
                check_output = err.check_output
                if isinstance(check_output, pd.DataFrame):
                    check_output = check_output.all(axis="columns")
                index_values = check_output.index[~check_output.astype(bool)]
            elif isinstance(err.failure_cases, pd.Series):
                index_values = err.failure_cases.index
            else:
                index_column = err.failure_cases["index"]
                if isinstance(check_obj.index, pd.MultiIndex):
                    index_tuples = [
                        _reported_label_keys(label)[0]
                        for label in index_column
                    ]
                    index_values = pd.MultiIndex.from_tuples(index_tuples)
                else:
                    index_values = index_column
                    # align dtypes so isin() matches (e.g. "1" vs 1)
                    try:
                        if hasattr(check_obj.index, "dtype") and hasattr(
                            index_values, "astype"
                        ):
                            index_values = index_values.astype(
                                check_obj.index.dtype
                            )
                    except (TypeError, ValueError):
                        pass

            mask = ~check_obj.index.isin(index_values)
            check_obj = check_obj.loc[mask]
        return check_obj

    def _collect_failing_ids(
        self, check_obj, error_handler
    ) -> np.ndarray | None:
        """Union of row ids that failed any collected schema error."""
        failing_ids = None
        for err in error_handler.schema_errors:
            ids = self._failing_ids_for_error(check_obj, err)
            if ids is None:
                continue
            failing_ids = (
                ids
                if failing_ids is None
                else np.concatenate([failing_ids, ids])
            )
        return failing_ids

    @staticmethod
    def _failing_ids_for_error(check_obj, err) -> np.ndarray | None:
        """Row ids that failed for a single collected schema error."""
        failure_cases = err.failure_cases
        if isinstance(failure_cases, str) or failure_cases is None:
            # A string failure case denotes a schema-level error.
            return None

        # Prefer the ids recorded when the check ran: they cover every
        # failing row even if the report was truncated.
        ids = get_failing_ids(failure_cases)
        if ids is not None:
            return ids

        # Fall back to the full check output when it aligns with the rows.
        ids = ids_from_mask(check_obj, output_failure_mask(err.check_output))
        if ids is not None:
            return ids

        # Last resort: resolve the reported labels to positions.
        return ids_from_labels(check_obj, failure_labels(failure_cases))
