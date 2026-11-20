"""Check backend for pandas."""

from collections.abc import Hashable
from functools import partial
from typing import Optional, Union, cast

import numpy as np
import pandas as pd

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.pandas.types import (
    GroupbyObject,
    is_bool,
    is_field,
    is_table,
    is_table_or_field,
)
from pandera.backends.base import BaseCheckBackend


class PandasCheckBackend(BaseCheckBackend):
    """Check backend for pandas."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj: Union[pd.Series, pd.DataFrame]):
        """Implements groupby behavior for check object."""
        assert self.check.groupby is not None, "Check.groupby must be set."
        if isinstance(self.check.groupby, (str, list)):
            return check_obj.groupby(self.check.groupby)
        return self.check.groupby(check_obj)

    def query(self, check_obj):
        """Implements querying behavior to produce subset of check object."""
        raise NotImplementedError

    def aggregate(self, check_obj):
        """Implements aggregation behavior for check object."""
        raise NotImplementedError

    @staticmethod
    def _format_groupby_input(
        groupby_obj: GroupbyObject,
        groups: list[Hashable] | None,
    ) -> dict[Hashable, pd.Series] | dict[Hashable, pd.DataFrame]:
        """Format groupby object into dict of groups to Series or DataFrame.

        :param groupby_obj: a pandas groupby object.
        :param groups: only include these groups in the output.
        :returns: dictionary mapping group names to Series or DataFrame.
        """
        # NOTE: this behavior should be deprecated such that the user deals with
        # pandas groupby objects instead of dicts.
        if groups is None:
            return {
                (
                    k
                    if isinstance(k, bool) or not isinstance(k, tuple)
                    else k[0]
                    if len(k) == 1
                    else k
                ): v
                for k, v in groupby_obj  # type: ignore[union-attr]
            }
        group_keys = {
            k if not isinstance(k, tuple) else k[0] if len(k) == 1 else k
            for k, _ in groupby_obj  # type: ignore[union-attr]
        }
        invalid_groups = [g for g in groups if g not in group_keys]
        if invalid_groups:
            raise KeyError(
                f"groups {invalid_groups} provided in `groups` argument not a "
                f"valid group key. Valid group keys: {group_keys}"
            )
        output = {}
        for group_key, group in groupby_obj:
            if isinstance(group_key, tuple) and len(group_key) == 1:
                group_key = group_key[0]
            if group_key in groups:
                output[group_key] = group

        return output  # type: ignore[return-value]

    def preprocess(
        self, check_obj, key
    ) -> (
        pd.Series
        | pd.DataFrame
        | dict[str, pd.Series]
        | dict[str, pd.DataFrame]
    ):
        """Preprocesses a check object before applying the check function."""
        # This handles the case of Series validation, which has no other context except
        # for the index to groupby on. Right now grouping by the index is not allowed.
        if is_field(check_obj):
            return self.preprocess_field(check_obj)
        elif is_table(check_obj) and key is None:
            return self.preprocess_table(check_obj)
        elif is_table(check_obj) and key is not None:
            return self.preprocess_table_with_key(check_obj, key)
        else:
            raise NotImplementedError

    def preprocess_field(
        self,
        check_obj,
    ) -> Union[pd.Series, dict[str, pd.Series]]:
        if self.check.groupby is None:
            if self.check.ignore_na and check_obj.hasnans:
                return check_obj.dropna()
            return check_obj
        return cast(
            dict[str, pd.Series],
            self._format_groupby_input(
                self.groupby(check_obj), self.check.groups
            ),
        )

    def preprocess_table_with_key(
        self,
        check_obj,
        key,
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        if self.check.groupby is None:
            if self.check.ignore_na and check_obj[key].hasnans:
                return check_obj[key].dropna()
            return check_obj[key]
        return cast(
            dict[str, pd.DataFrame],
            self._format_groupby_input(
                self.groupby(check_obj)[key], self.check.groups
            ),
        )

    def preprocess_table(
        self,
        check_obj,
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        if self.check.groupby is None:
            return check_obj
        return cast(
            dict[str, pd.DataFrame],
            self._format_groupby_input(
                self.groupby(check_obj), self.check.groups
            ),
        )

    def apply(self, check_obj):
        """Apply the check function to a check object."""
        if isinstance(check_obj, dict):
            apply_fn = self.apply_dict
        elif is_field(check_obj):
            apply_fn = self.apply_field
        elif is_table(check_obj):
            apply_fn = self.apply_table
        else:
            raise NotImplementedError
        return apply_fn(check_obj)

    def apply_dict(self, check_obj: dict):
        return self.check_fn(check_obj)

    def apply_field(self, check_obj):
        if self.check.element_wise:
            return check_obj.map(self.check_fn)
        return self.check_fn(check_obj)

    def apply_table(self, check_obj):
        if self.check.element_wise:
            return check_obj.apply(self.check_fn, axis=1)
        return self.check_fn(check_obj)

    def postprocess(self, check_obj, check_output):
        """Postprocesses the result of applying the check function."""
        if is_field(check_obj) and is_field(check_output):
            postprocess_fn = self.postprocess_field
        elif is_table(check_obj) and is_table(check_output):
            postprocess_fn = self.postprocess_table
        elif is_table(check_obj) and is_field(check_output):
            postprocess_fn = self.postprocess_table_with_field_output
        elif is_table_or_field(check_obj) and is_bool(check_output):
            postprocess_fn = self.postprocess_table_or_field_with_bool_output
        elif isinstance(check_obj, dict) and is_field(check_output):
            postprocess_fn = self.postprocess_dict_with_field_output
        elif is_bool(check_output):
            postprocess_fn = self.postprocess_bool
        else:
            raise NotImplementedError
        return postprocess_fn(check_obj, check_output)

    def postprocess_bool(
        self,
        check_obj,
        check_output,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        return CheckResult(
            check_output=check_output,
            check_passed=check_output,
            checked_object=check_obj,
            failure_cases=None,
        )

    def _get_series_failure_cases(
        self,
        check_obj,
        check_output: pd.Series,
    ) -> pd.Series | None:
        if not check_obj.index.equals(check_output.index):
            return None

        if check_output.all():
            return None

        if check_output.dtype != bool:
            check_output = check_output.astype(bool)

        failure_cases = check_obj[~check_output]
        if not failure_cases.empty and self.check.n_failure_cases is not None:
            # NOTE: this is a hack to support pyspark.pandas and modin, since you
            # can't use groupby on a dataframe with another dataframe
            if type(failure_cases).__module__.startswith(
                ("pyspark.pandas", "modin.pandas")
            ):
                failure_cases = (
                    failure_cases.rename("failure_cases")
                    .to_frame()
                    .assign(check_output=check_output)
                    .groupby("check_output")
                    .head(self.check.n_failure_cases)["failure_cases"]
                )
            else:
                failure_cases = failure_cases.groupby(check_output).head(
                    self.check.n_failure_cases
                )
        return failure_cases

    def postprocess_field(
        self,
        check_obj,
        check_output,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        if check_output.empty and check_output.dtype != bool:
            check_output = check_output.astype(bool)

        return CheckResult(
            check_output,
            check_output.all(),
            check_obj,
            self._get_series_failure_cases(check_obj, check_output),
        )

    def postprocess_table_with_field_output(
        self,
        check_obj,
        check_output,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        if check_output.empty and check_output.dtype != bool:
            check_output = check_output.astype(bool)

        if check_obj.index.equals(check_output.index) and self.check.ignore_na:
            check_output = check_output | check_obj.isna().all(axis="columns")
        return CheckResult(
            check_output,
            check_output.all(),
            check_obj,
            self._get_series_failure_cases(check_obj, check_output),
        )

    def postprocess_table(
        self,
        check_obj,
        check_output,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        assert check_obj.shape == check_output.shape

        for col, dtype in check_output.dtypes.items():
            if check_output[col].empty and dtype != bool:
                check_output[col] = check_output[col].astype(bool)

        if check_obj.index.equals(check_output.index) and self.check.ignore_na:
            check_output = check_output | check_obj.isna()

        return CheckResult(
            check_output,
            check_output.all(axis=None),  # type: ignore
            check_obj,
            self._table_failure_cases(check_obj, check_output),
        )

    @staticmethod
    def _can_scan_table_failure_cases(check_obj, check_output) -> bool:
        """Whether failure cases can be built without a full report.

        The scanning path walks the per-cell failure mask directly, which is
        only equivalent to the reporting path when both frames are aligned
        and boolean. Malformed check outputs (misaligned axes, non-boolean
        outputs, duplicate column labels, MultiIndex rows) and the pandas-like
        backends (modin, pyspark.pandas) keep using the reporting path so its
        existing behavior is preserved.
        """
        if not check_obj.columns.is_unique:
            return False
        if isinstance(check_obj.index, pd.MultiIndex):
            return False
        if not check_obj.index.equals(check_output.index):
            return False
        if not check_obj.columns.equals(check_output.columns):
            return False
        if any(dtype != bool for dtype in check_output.dtypes):
            return False
        # Only native pandas frames are scanned: subclasses and the
        # pandas-like backends (modin, pyspark.pandas, geopandas) keep the
        # reporting path so their behavior is untouched.
        return type(check_obj) is pd.DataFrame and (
            type(check_output) is pd.DataFrame
        )

    def _table_failure_cases(self, check_obj, check_output) -> pd.DataFrame:
        """Build the table-level failure-case report.

        When the check output is aligned and boolean, failures are read
        straight from the per-cell mask so that only the reported rows are
        materialized: intermediate per-column frames for every failing cell
        are never built, and with a ``n_failure_cases`` limit the scan stops
        as soon as enough unique failure cases are collected.
        """
        if self._can_scan_table_failure_cases(check_obj, check_output):
            return self._scan_table_failure_cases(check_obj, check_output)
        return self._build_table_failure_cases(check_obj, check_output)

    def _build_table_failure_cases(
        self, check_obj, check_output
    ) -> pd.DataFrame:
        """Build the full failure-case report before limiting it."""
        # collect failure cases across all columns. False values in check_output
        # are nulls.
        select_failure_cases = check_obj[~check_output]
        failure_cases_list: list[pd.DataFrame] = []
        for col in select_failure_cases.columns:
            cases = select_failure_cases[col].rename("failure_case").dropna()
            if len(cases) == 0:
                continue
            failure_cases_list.append(
                cases.to_frame()
                .assign(column=col)
                .rename_axis("index")
                .reset_index()
            )

        if failure_cases_list:
            failure_cases = pd.concat(failure_cases_list, axis=0)
            # convert to a dataframe where each row is a failure case at
            # a particular index, and failure case values are dictionaries
            # indicating which column and value failed in that row.
            failure_cases = (
                failure_cases.set_index("column")
                .groupby("index")
                .agg(lambda df: df.to_dict())
            )
        else:
            failure_cases = pd.DataFrame(columns=["index", "failure_case"])

        if not failure_cases.empty and self.check.n_failure_cases is not None:
            failure_cases = failure_cases.drop_duplicates().head(
                self.check.n_failure_cases
            )
        return failure_cases

    @staticmethod
    def _masked_column_dtype(series: pd.Series, masked) -> object:
        """Dtype of ``series`` after masking out non-failing cells."""
        if masked.all() or len(series) == 0:
            return series.dtype
        return (
            series.iloc[:1]
            .where(pd.Series([False], index=series.index[:1]))
            .dtype
        )

    @staticmethod
    def _failure_case_index(labels, index: pd.Index) -> pd.Index:
        """Rebuild the report index with the original labels and dtype."""
        if isinstance(index, pd.CategoricalIndex):
            return pd.CategoricalIndex(
                labels,
                categories=index.categories,
                ordered=index.ordered,
                name="index",
            )
        return pd.Index(labels, dtype=index.dtype, name="index")

    @staticmethod
    def _failure_case_frame(labels, payloads, index: pd.Index) -> pd.DataFrame:
        values = np.empty(len(payloads), dtype=object)
        for position, payload in enumerate(payloads):
            values[position] = payload
        result_index = PandasCheckBackend._failure_case_index(labels, index)
        return pd.DataFrame(
            {
                "failure_case": pd.Series(
                    values, index=result_index, dtype=object
                )
            }
        )

    def _scan_table_failure_cases(
        self, check_obj, check_output
    ) -> pd.DataFrame:
        """Read failure cases from the mask, materializing only reported rows."""
        index = check_obj.index
        failure_masks: dict[object, np.ndarray] = {}
        column_series: dict[object, pd.Series] = {}
        masked_dtypes: dict[object, object] = {}
        row_has_failure = np.zeros(len(check_obj), dtype=bool)

        for col in check_obj.columns:
            series = check_obj[col]
            masked = ~check_output[col].to_numpy()
            failures = masked & series.notna().to_numpy()
            if not failures.any():
                continue
            failure_masks[col] = failures
            column_series[col] = series
            masked_dtypes[col] = self._masked_column_dtype(series, masked)
            row_has_failure |= failures

        if not failure_masks:
            return pd.DataFrame(columns=["index", "failure_case"])

        # The per-column values share one dtype in the report, exactly as the
        # concatenated failure-case frame does in ``_build_table_failure_cases``.
        common_dtype = pd.concat(
            [pd.Series([], dtype=masked_dtypes[col]) for col in failure_masks]
        ).dtype

        positions = np.flatnonzero(row_has_failure)
        labels_index = check_obj.index.take(positions)
        if labels_index.isna().all():
            # ``groupby`` drops NA labels, so no group is ever reported. The
            # empty report keeps the concatenated dtype, as before.
            return pd.DataFrame(
                {
                    "failure_case": pd.Series([], dtype=common_dtype),
                },
                index=self._failure_case_index([], check_obj.index),
            )
        grouper = pd.Series(positions, index=labels_index)
        n_failure_cases = self.check.n_failure_cases
        scanners = (
            failure_masks,
            column_series,
            masked_dtypes,
            common_dtype,
        )

        if n_failure_cases is None:
            # No limit: every failing row is reported, without deduplication.
            labels, payloads = self._collect_failure_cases(grouper, *scanners)
        elif n_failure_cases < 0:
            # ``head`` with a negative limit drops the tail of the deduplicated
            # report, so the full set of unique cases has to be known first.
            labels, payloads = self._collect_failure_cases(
                grouper, *scanners, unique=True
            )
            labels = labels[:n_failure_cases]
            payloads = payloads[:n_failure_cases]
        else:
            labels, payloads = self._collect_failure_cases(
                grouper, *scanners, unique=True, limit=n_failure_cases
            )
        return self._failure_case_frame(labels, payloads, check_obj.index)

    # Rows are scanned and de-duplicated in bounded batches so that a small
    # ``n_failure_cases`` never materializes a report for every failing cell.
    _FAILURE_CASE_CHUNK_SIZE = 512

    def _collect_failure_cases(
        self,
        grouper,
        failure_masks,
        column_series,
        masked_dtypes,
        common_dtype,
        unique: bool = False,
        limit: int | None = None,
    ) -> tuple[list, list[dict]]:
        """Reported failure cases, one dictionary per row label.

        With ``unique``, payloads are de-duplicated exactly as
        ``drop_duplicates`` would, in bounded batches. Once ``limit`` unique
        cases are known the scan stops, so rows past the report cap never
        become payloads.
        """
        labels: list = []
        payloads: list[dict] = []
        chunk_labels: list = []
        chunk_values: list[dict] = []
        chunk_size = self._FAILURE_CASE_CHUNK_SIZE
        done = False

        for label, group in grouper.groupby(level=0, sort=True, dropna=True):
            pos = group.to_numpy()
            values = {}
            for col, failures in failure_masks.items():
                row_positions = pos[failures[pos]]
                if row_positions.size:
                    values[col] = column_series[col].iat[
                        int(row_positions[-1])
                    ]
            chunk_labels.append(label)
            chunk_values.append(values)
            if len(chunk_labels) >= chunk_size:
                labels, payloads, done = self._merge_failure_cases(
                    labels,
                    payloads,
                    chunk_labels,
                    chunk_values,
                    masked_dtypes,
                    common_dtype,
                    unique,
                    limit,
                )
                chunk_labels, chunk_values = [], []
                if done:
                    break
        if not done and chunk_values:
            labels, payloads, done = self._merge_failure_cases(
                labels,
                payloads,
                chunk_labels,
                chunk_values,
                masked_dtypes,
                common_dtype,
                unique,
                limit,
            )
        if limit is not None:
            labels, payloads = labels[:limit], payloads[:limit]
        return labels, payloads

    def _merge_failure_cases(
        self,
        labels: list,
        payloads: list[dict],
        chunk_labels: list,
        chunk_values: list[dict],
        masked_dtypes,
        common_dtype,
        unique: bool,
        limit: int | None,
    ) -> tuple[list, list[dict], bool]:
        """Box and merge one batch, keeping only the required unique cases."""
        boxed = self._box_failure_cases(
            chunk_values, masked_dtypes, common_dtype
        )
        labels = labels + chunk_labels
        payloads = payloads + boxed
        if unique:
            keep = ~pd.Series(payloads, dtype=object).duplicated()
            labels = [
                label for label, keep_row in zip(labels, keep) if keep_row
            ]
            payloads = [
                payload
                for payload, keep_row in zip(payloads, keep)
                if keep_row
            ]
        done = limit is not None and len(payloads) >= limit
        if done:
            labels, payloads = labels[:limit], payloads[:limit]
        return labels, payloads, done

    @staticmethod
    def _box_failure_cases(
        chunk_values, masked_dtypes, common_dtype
    ) -> list[dict]:
        """Box each raw failure value the way ``Series.to_dict`` would."""
        payloads: list[dict] = [{} for _ in chunk_values]
        for col, masked_dtype in masked_dtypes.items():
            entries = [
                (row, values[col])
                for row, values in enumerate(chunk_values)
                if col in values
            ]
            if not entries:
                continue
            boxed = (
                pd.Series([value for _, value in entries], dtype=masked_dtype)
                .astype(common_dtype)
                .to_dict()
            )
            for offset, (row, _) in enumerate(entries):
                payloads[row][col] = boxed[offset]
        return payloads

    def postprocess_table_or_field_with_bool_output(
        self,
        check_obj,
        check_output,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        check_output = bool(check_output)
        return CheckResult(
            check_output,
            check_output,
            check_obj,
            None,
        )

    def postprocess_dict_with_field_output(
        self,
        check_obj,
        check_output,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        return CheckResult(
            check_output,
            check_output.all(),
            check_obj,
            None,
        )

    def __call__(
        self,
        check_obj: Union[pd.Series, pd.DataFrame],
        key: str | None = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        check_output = self.apply(check_obj)
        return self.postprocess(check_obj, check_output)
