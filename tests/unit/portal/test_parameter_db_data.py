"""Tests for the Parameter Database data/controller layer."""

from __future__ import annotations

from larvaworld.lib import reg
from larvaworld.portal.parameter_database import parameter_db_data


def test_build_parameter_table_df_row_count_matches_dict() -> None:
    df = parameter_db_data.build_parameter_table_df()
    assert len(df) == len(reg.par.dict)
    assert list(df.columns) == [label for _, label in parameter_db_data.TABLE_COLUMNS]


def test_table_columns_follow_larvaworldparam_precedence() -> None:
    attrs = [attr for attr, _ in parameter_db_data.TABLE_COLUMNS]
    non_category = [a for a in attrs if a != "category"]
    precedences = [parameter_db_data._attr_precedence(a) for a in non_category]
    # Non-None precedences (ties allowed) must be non-increasing; None-
    # precedence attrs (e.g. codename) sort after all ranked ones.
    ranked = [p for p in precedences if p is not None]
    assert ranked == sorted(ranked, reverse=True)
    assert attrs[-1] == "category"


def test_table_columns_include_all_attrs_including_hidden_ones() -> None:
    # All attributes are offered as toggle-able columns, including
    # negative-precedence ones -- DEFAULT_HIDDEN_COLUMNS (not TABLE_COLUMNS)
    # is what determines which start unchecked.
    attrs = {attr for attr, _ in parameter_db_data.TABLE_COLUMNS}
    assert {"flatname", "func", "required_ks"} <= attrs


def test_default_hidden_columns_are_below_visibility_threshold() -> None:
    attr_by_label = {label: attr for attr, label in parameter_db_data.TABLE_COLUMNS}
    threshold = parameter_db_data._TABLE_VISIBILITY_PRECEDENCE_THRESHOLD
    for label in parameter_db_data.DEFAULT_HIDDEN_COLUMNS:
        attr = attr_by_label[label]
        prec = parameter_db_data._attr_precedence(attr)
        assert (prec or 0) < threshold

    shown_attrs = {
        attr
        for attr, label in parameter_db_data.TABLE_COLUMNS
        if attr != "category" and label not in parameter_db_data.DEFAULT_HIDDEN_COLUMNS
    }
    for attr in shown_attrs:
        assert (parameter_db_data._attr_precedence(attr) or 0) >= threshold


def test_table_df_has_default_value_column_not_description() -> None:
    df = parameter_db_data.build_parameter_table_df()
    reg.par.update_kdict(["t"])
    lp = reg.par.kdict["t"]
    row = df[df["Key"] == "t"].iloc[0]
    assert row["Default value"] == str(lp.v0)
    assert "Description" not in df.columns
