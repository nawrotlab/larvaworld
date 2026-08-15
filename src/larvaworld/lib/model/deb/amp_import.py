"""
Import a DEB parameter set from an Add-my-Pet (AmP) results page.

AmP publishes each species entry as an HTML results page containing a parameter
table and a data/predictions table. This module converts such a page into the JSON
form that :meth:`~larvaworld.lib.model.deb.deb_equations.DEBPars.from_amp_json`
consumes, which is the supported path for adopting a newer AmP export.

The page is read from a **local file**; nothing here contacts the network. Download
the results page yourself, then point this importer at it.

Expected structure
------------------
1. A ``<title>`` holding the species name, e.g. ``<title>Drosophila_melanogaster</title>``.
2. A parameter table with columns ``symbol | unit | value | free | description``,
   identified by ``id="par"``. If that id is absent the first table is used.
3. A data/predictions table with columns ``data | prd | RE | symbol | units |
   description``, identified by ``id="prd"``.
4. An ``<h2>`` like ``abp parameters at 20.0 degC``; its leading token becomes
   ``metadata["typified_model"]``.

Output
------
``{"metadata": {"species", "typified_model"},
   "parameters": [{"symbol", "unit", "value", "free", "description"}, ...],
   "data_predictions": [{"data", "prd", "RE", "symbol", "unit", "description"}, ...]}``

Command line
------------
::

    python -m larvaworld.lib.model.deb.amp_import path/to/Species_res.html \
        --out src/larvaworld/lib/model/deb/models/amp/Species.json

Ported from ``retrieve_pars_from_html_to_json.py`` in the external
``matlab_to_python`` reference folder.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

import argparse
import json

__all__: list[str] = [
    "retrieve_pars_from_html",
    "ParameterTableParser",
    "MetadataParser",
]


class ParameterTableParser(HTMLParser):
    """Collect the rows of one HTML table, selected by ``id`` (default ``"par"``)."""

    def __init__(self, table_id: Optional[str] = "par") -> None:
        super().__init__()
        self.table_id = table_id
        self._in_table = False
        self._in_tr = False
        self._in_cell = False
        self._cell_chunks: list[str] = []
        self._current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag == "table":
            attr = {k.lower(): (v or "") for k, v in attrs}
            if self.table_id is None or attr.get("id") == self.table_id:
                self._in_table = True
        if self._in_table and tag == "tr":
            self._in_tr = True
            self._current_row = []
        if self._in_table and self._in_tr and tag in ("td", "th"):
            self._in_cell = True
            self._cell_chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._in_table and self._in_tr and tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append("".join(self._cell_chunks).strip())
            self._cell_chunks = []
        if self._in_table and tag == "tr":
            self._in_tr = False
            if self._current_row:
                self.rows.append(self._current_row)
                self._current_row = []
        if self._in_table and tag == "table":
            self._in_table = False


class MetadataParser(HTMLParser):
    """Extract the species from ``<title>`` and the model type from the ``<h2>``."""

    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._in_h2 = False
        self._title_chunks: list[str] = []
        self._h2_chunks: list[str] = []
        self.species: Optional[str] = None
        self.typified_model: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "title":
            self._in_title, self._title_chunks = True, []
        elif tag.lower() == "h2":
            self._in_h2, self._h2_chunks = True, []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_chunks.append(data)
        elif self._in_h2:
            self._h2_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
            title = "".join(self._title_chunks).strip()
            if title:
                self.species = title
        elif tag.lower() == "h2":
            self._in_h2 = False
            text = "".join(self._h2_chunks).strip()
            if text and "parameters at" in text.lower():
                token = text.split()[0]
                if token:
                    self.typified_model = token


def _coerce_value(text: str) -> Any:
    """Parse numeric text as float where possible, otherwise return it unchanged."""
    t = text.strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        try:
            return float(t.replace(" ", ""))
        except ValueError:
            return t


def _coerce_free(text: str) -> Optional[int]:
    """Normalise the ``free`` column to 0 or 1."""
    val = _coerce_value(text)
    if val is None:
        return None
    try:
        return 1 if float(val) != 0 else 0
    except (TypeError, ValueError):
        return None


def _rows_to_params(rows: list[list[str]]) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for row in rows:
        if not row or row[0].strip().lower() == "symbol" or len(row) < 5:
            continue
        params.append(
            {
                "symbol": row[0].strip(),
                "unit": row[1].strip(),
                "value": _coerce_value(row[2]),
                "free": _coerce_free(row[3]),
                "description": row[4].strip(),
            }
        )
    return params


def _rows_to_data_predictions(rows: list[list[str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        if not row or row[0].strip().lower() == "data" or len(row) < 4:
            continue

        if len(row) == 4:
            # colspan collapsed data/prd and units/description into single cells
            data_text, prd_text, re_text = row[0].strip(), "", row[1].strip()
            symbol, unit_text, description = row[2].strip(), "", row[3].strip()
        elif len(row) == 5:
            data_text, prd_text, re_text = (c.strip() for c in row[:3])
            symbol, unit_text, description = row[3].strip(), "", row[4].strip()
        else:
            data_text, prd_text, re_text = (c.strip() for c in row[:3])
            symbol, unit_text, description = (c.strip() for c in row[3:6])

        items.append(
            {
                "data": _coerce_value(data_text),
                "prd": _coerce_value(prd_text) if prd_text else None,
                "RE": _coerce_value(re_text),
                "symbol": symbol,
                "unit": unit_text,
                "description": description,
            }
        )
    return items


def retrieve_pars_from_html(
    html_path: str, table_id: Optional[str] = "par"
) -> dict[str, Any]:
    """
    Parse a local AmP results page into the JSON payload ``DEBPars.from_amp_json`` reads.

    Parameters
    ----------
    html_path : path to a downloaded ``*_res.html`` page
    table_id : id of the parameter table; falls back to the first table if absent

    Returns
    -------
    dict with ``metadata``, ``parameters`` and ``data_predictions``.
    """
    html_text = Path(html_path).read_text(encoding="utf-8", errors="replace")

    meta = MetadataParser()
    meta.feed(html_text)

    par = ParameterTableParser(table_id=table_id)
    par.feed(html_text)
    if not par.rows and table_id is not None:
        par = ParameterTableParser(table_id=None)
        par.feed(html_text)

    prd = ParameterTableParser(table_id="prd")
    prd.feed(html_text)

    return {
        "metadata": {
            "species": meta.species,
            "typified_model": meta.typified_model,
        },
        "parameters": _rows_to_params(par.rows),
        "data_predictions": _rows_to_data_predictions(prd.rows),
    }


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point; see the module docstring for usage."""
    ap = argparse.ArgumentParser(
        description="Convert a local AmP results page into a DEB parameter JSON file."
    )
    ap.add_argument("html_path", help="Path to a downloaded *_res.html page")
    ap.add_argument(
        "--table-id", default="par", help="Parameter table id (default: par)"
    )
    ap.add_argument(
        "--out",
        help="Output JSON path; defaults to <species>.json beside this module's "
        "models/amp/ folder.",
    )
    args = ap.parse_args(argv)

    payload = retrieve_pars_from_html(args.html_path, table_id=args.table_id)
    species = payload["metadata"].get("species")

    if args.out:
        out_path = Path(args.out)
    else:
        if not species:
            raise ValueError(
                "Could not read the species from <title>; pass --out explicitly."
            )
        safe = species.strip().replace("/", "_").replace("\\", "_")
        out_path = Path(__file__).resolve().parent / "models" / "amp" / f"{safe}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(payload['parameters'])} parameters to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
