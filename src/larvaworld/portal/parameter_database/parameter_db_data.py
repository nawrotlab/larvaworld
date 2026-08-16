"""Data/controller layer for the Parameter Database popup.

Queries `larvaworld.lib.reg.par` (the global `ParamRegistry`) and returns
plain data structures (DataFrames, dicts) for the Panel UI layer to render.
No Panel/Bokeh imports here.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from larvaworld.lib import reg
from larvaworld.lib.reg.data_aux import LarvaworldParam
from larvaworld.lib.reg.parDB import ParamRegistry

__all__: list[str] = [
    "TABLE_COLUMNS",
    "DEFAULT_HIDDEN_COLUMNS",
    "build_parameter_table_df",
]

#: "unit" and "v0" are @property values on LarvaworldParam (`self.unit`,
#: `self.param.v.default`), not declared `param.Parameter`s -- see
#: `larvaworld.lib.reg.data_aux.LarvaworldParam`/`get_LarvaworldParam` --
#: so they aren't discoverable via `LarvaworldParam.param` below and need
#: their own header text here instead of a `.label` to read.
_SPECIAL_COLUMN_LABELS: dict[str, str] = {
    "unit": "Unit",
    "v0": "Default value",
}

#: Attribute names offered as candidate columns for the main parameter
#: table: every LarvaworldParam-declared attribute (discovered directly
#: from the class, so this doesn't fall out of sync with it), plus the two
#: special pseudo-columns above. Column headers aren't hardcoded here: for
#: every attribute except "unit"/"v0", the header comes straight from that
#: attribute's `label=` on `LarvaworldParam` (see
#: `larvaworld.lib.reg.data_aux`) via `_attr_label`, so there's a single
#: source of truth for it rather than a second copy in this module.
_ATTR_CANDIDATE_ATTRS: list[str] = [
    p for p in LarvaworldParam.param if p != "name"
] + list(_SPECIAL_COLUMN_LABELS)

#: `u` (the source of the "unit" column) is only declared on the dynamically
#: created subclass built by `reg.get_LarvaworldParam`, not on the base
#: `LarvaworldParam` class, so its precedence can't be read off the class
#: here. Mirrors the precedence set there.
_UNIT_PRECEDENCE = 8

#: "v0" is a @property on LarvaworldParam (`self.param.v.default`), not a
#: declared param.Parameter, so it has no `.precedence` attribute either.
_DEFAULT_VALUE_PRECEDENCE = 8


def _attr_label(attr: str) -> str:
    if attr in _SPECIAL_COLUMN_LABELS:
        return _SPECIAL_COLUMN_LABELS[attr]
    return LarvaworldParam.param[attr].label


def _attr_precedence(attr: str) -> Optional[float]:
    if attr == "unit":
        return _UNIT_PRECEDENCE
    if attr == "v0":
        return _DEFAULT_VALUE_PRECEDENCE
    return LarvaworldParam.param[attr].precedence


def _ordered_attr_columns() -> list[tuple[str, str]]:
    """All attribute columns, following `LarvaworldParam` precedence
    (highest first); attributes with no precedence set are shown last, in
    declaration order. Negative-precedence attributes are included too (see
    DEFAULT_HIDDEN_COLUMNS) rather than dropped, so they remain available as
    opt-in columns.
    """
    rows = [
        (attr, _attr_label(attr), _attr_precedence(attr))
        for attr in _ATTR_CANDIDATE_ATTRS
    ]
    rows.sort(key=lambda row: (row[2] is None, -(row[2] if row[2] is not None else 0)))
    return [(attr, label) for attr, label, _ in rows]


#: Ordered (attribute, column header) pairs for the main parameter table.
#: "category" is a derived column (not a LarvaworldParam attribute; see
#: ParamClass.category_of) and is always appended last.
TABLE_COLUMNS: list[tuple[str, str]] = _ordered_attr_columns() + [
    ("category", "Category")
]

#: Precedence threshold for a column to start visible in the table: >=3 is
#: shown/checked by default, everything else starts hidden/unchecked.
#: "category" (a derived column with no LarvaworldParam precedence) always
#: stays visible, as before.
_TABLE_VISIBILITY_PRECEDENCE_THRESHOLD = 3

#: Column headers (from TABLE_COLUMNS) hidden by default: attributes whose
#: precedence is below _TABLE_VISIBILITY_PRECEDENCE_THRESHOLD.
DEFAULT_HIDDEN_COLUMNS: list[str] = [
    label
    for attr, label in TABLE_COLUMNS
    if attr != "category"
    and (_attr_precedence(attr) or 0) < _TABLE_VISIBILITY_PRECEDENCE_THRESHOLD
]


def _dtype_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", str(dtype))


def _column_value(prepar: Any, attr: str) -> Any:
    """The table-cell value for one TABLE_COLUMNS attribute, read off a
    prepared-spec AttrDict (`par.dict[k]`) -- most attributes are used
    as-is; dtype/unit/default-value/required-keys/func are formatted for
    display. `prepar`'s keys mirror `LarvaworldParam`'s real attribute
    names (see `prepare_LarvaworldParam`), so this reads identically off
    either one."""
    if attr == "dtype":
        return _dtype_name(prepar.dtype)
    if attr == "unit":
        return str(prepar.u)
    if attr == "v0":
        return "" if prepar.v0 is None else str(prepar.v0)
    if attr == "flatname":
        return prepar.flatname or ""
    if attr == "required_ks":
        return ", ".join(prepar.required_ks) if prepar.required_ks else ""
    if attr == "func":
        func = prepar.func
        return f"{func.__module__}.{func.__qualname__}" if func is not None else ""
    return getattr(prepar, attr)


def build_parameter_table_df(par: Optional[ParamRegistry] = None) -> pd.DataFrame:
    """
    Build a DataFrame with one row per registered parameter and columns per
    TABLE_COLUMNS. Reads from `par.dict` (the prepared parameter specs), not
    `par.kdict`, so it never triggers per-parameter LarvaworldParam
    instantiation.
    """
    par = par if par is not None else reg.par
    attrs = [attr for attr, _ in TABLE_COLUMNS if attr != "category"]
    rows = []
    for k, prepar in par.dict.items():
        row = {attr: _column_value(prepar, attr) for attr in attrs}
        row["category"] = par.category_of(k)
        rows.append(row)
    columns = [attr for attr, _ in TABLE_COLUMNS]
    df = pd.DataFrame(rows, columns=columns)
    df = df.rename(columns=dict(TABLE_COLUMNS))
    return df
