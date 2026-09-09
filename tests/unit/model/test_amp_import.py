"""
Tests for the Add-my-Pet results-page importer.

The importer converts a locally downloaded AmP HTML results page into the JSON
payload that ``DEBPars.from_amp_json`` consumes. These tests use a synthetic page
so they are hermetic; an optional test runs against the real page when it happens
to be available.
"""

from __future__ import annotations

import json
import os

import pytest

from larvaworld.lib.model.deb import amp_import
from larvaworld.lib.model.deb import deb_equations as de
from larvaworld.lib.model.deb import rover_sitter_model as rs

#: A miniature AmP results page. The prd table deliberately mixes 6-, 5- and
#: 4-cell rows to exercise the colspan handling of the real pages.
FIXTURE_HTML = """<html>
<head><title>Testus exampleii</title></head>
<body>
<h1>Ignore me</h1>
<h2>abj parameters at 20.0 degC</h2>
<table id="par">
  <tr><th>symbol</th><th>unit</th><th>value</th><th>free</th><th>description</th></tr>
  <tr><td>T_A</td><td>K</td><td>8000</td><td>1</td><td>Arrhenius temperature</td></tr>
  <tr><td>z</td><td>-</td><td>0.5054</td><td>1</td><td>zoom factor</td></tr>
  <tr><td>kap</td><td>-</td><td>0.8024</td><td>0</td><td>allocation fraction to soma</td></tr>
  <tr><td>notanumber</td><td>-</td><td>n/a</td><td></td><td>unparseable value</td></tr>
</table>
<h2>Data and predictions</h2>
<table id="prd">
  <tr><th>data</th><th>prd</th><th>RE</th><th>symbol</th><th>units</th><th>description</th></tr>
  <tr><td>0.7</td><td>0.7236</td><td>0.03378</td><td>ab</td><td>d</td><td>age at birth</td></tr>
  <tr><td>0.06</td><td>0.05788</td><td>0.03541</td><td>Lb</td><td>cm</td></tr>
  <tr><td>11.2</td><td>0.1412</td><td>Ri</td><td>mean daily fecundity</td></tr>
</table>
</body></html>
"""


@pytest.fixture
def page(tmp_path) -> str:
    path = tmp_path / "Testus_exampleii_res.html"
    path.write_text(FIXTURE_HTML, encoding="utf-8")
    return str(path)


def test_metadata_is_read_from_title_and_header(page: str) -> None:
    payload = amp_import.retrieve_pars_from_html(page)
    assert payload["metadata"] == {
        "species": "Testus exampleii",
        "typified_model": "abj",
    }


def test_parameters_are_parsed(page: str) -> None:
    params = {
        e["symbol"]: e for e in amp_import.retrieve_pars_from_html(page)["parameters"]
    }
    assert params["T_A"]["value"] == 8000.0
    assert params["T_A"]["free"] == 1
    assert params["T_A"]["unit"] == "K"
    assert params["T_A"]["description"] == "Arrhenius temperature"
    assert params["kap"]["free"] == 0
    # the header row must not become a parameter
    assert "symbol" not in params


def test_unparseable_values_are_kept_as_text(page: str) -> None:
    params = {
        e["symbol"]: e for e in amp_import.retrieve_pars_from_html(page)["parameters"]
    }
    assert params["notanumber"]["value"] == "n/a"
    assert params["notanumber"]["free"] is None


def test_data_predictions_handle_ragged_rows(page: str) -> None:
    preds = {
        e["symbol"]: e
        for e in amp_import.retrieve_pars_from_html(page)["data_predictions"]
    }
    assert preds["ab"]["prd"] == 0.7236
    assert preds["ab"]["unit"] == "d"
    assert preds["Lb"]["prd"] == 0.05788  # 5-cell row, unit dropped
    assert preds["Ri"]["prd"] is None  # 4-cell row, data/prd collapsed
    assert preds["Ri"]["RE"] == 0.1412


def test_falls_back_to_the_first_table_when_the_id_is_absent(page: str) -> None:
    payload = amp_import.retrieve_pars_from_html(page, table_id="nosuchid")
    assert [e["symbol"] for e in payload["parameters"]][:2] == ["T_A", "z"]


def test_output_feeds_amp_predictions_and_DEBPars(page: str, tmp_path) -> None:
    """The payload must be consumable by the loaders it exists to serve."""
    payload = amp_import.retrieve_pars_from_html(page)
    out = tmp_path / "Testus.json"
    out.write_text(json.dumps(payload), encoding="utf-8")

    assert de.amp_predictions(str(out))["ab"] == 0.7236
    # from_amp_json takes only the symbols DEBPars declares and ignores the rest
    pars = de.DEBPars.from_amp_json(str(out))
    assert pars.T_A == 8000.0
    assert pars.z == 0.5054
    assert pars.metadata["species"] == "Testus exampleii"


def test_cli_writes_json(page: str, tmp_path) -> None:
    out = tmp_path / "out.json"
    assert amp_import.main([page, "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metadata"]["species"] == "Testus exampleii"
    assert len(payload["parameters"]) == 4


# ---------------------------------------------------------------------------
# Optional: the real AmP page, when present on this machine
# ---------------------------------------------------------------------------

REAL_PAGE = (
    r"G:\Το Drive μου\DEB projects\Drosophila DEB model\code"
    r"\Drosophilla_DEB_Evridiki\Drosophila_melanogaster_res.html"
)


@pytest.mark.skipif(
    not os.path.exists(REAL_PAGE),
    reason="the reference AmP page is not on this machine",
)
def test_real_page_matches_the_vendored_schema() -> None:
    """
    Parsing the published Drosophila page must reproduce the structure of the
    vendored export: same rows, symbols, units, descriptions and free flags.

    Values are *not* compared. The page on disk is an earlier AmP run than the
    vendored JSON (it still carries a positive kap_V), so the numbers legitimately
    differ while the schema does not.
    """
    got = amp_import.retrieve_pars_from_html(REAL_PAGE)
    with open(rs.DROSOPHILA_AMP_JSON, encoding="utf-8") as fh:
        ref = json.load(fh)

    assert got["metadata"] == ref["metadata"]
    assert len(got["parameters"]) == len(ref["parameters"])
    assert len(got["data_predictions"]) == len(ref["data_predictions"])
    for a, b in zip(got["parameters"], ref["parameters"]):
        assert (a["symbol"], a["unit"], a["free"], a["description"]) == (
            b["symbol"],
            b["unit"],
            b["free"],
            b["description"],
        )
    assert all(isinstance(e["value"], float) for e in got["parameters"])
