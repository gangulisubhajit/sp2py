"""
Schema / table-detail ingestion.

Users can optionally supply an Excel workbook describing their tables so
the conversion engine can map SQL types accurately. If nothing is
supplied, we fall back to a small bundled "dummy" schema so the app is
usable out of the box.
"""

from __future__ import annotations

import io

import pandas as pd

EXPECTED_COLUMNS = ["table_name", "column_name", "data_type", "nullable", "primary_key", "foreign_key"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[EXPECTED_COLUMNS]


def load_schema_from_excel(file_bytes: bytes) -> pd.DataFrame:
    """Parse an uploaded .xlsx workbook of table/column details.

    Expected columns (case-insensitive, order-independent):
    table_name, column_name, data_type, nullable, primary_key, foreign_key

    Extra columns are ignored; missing ones are filled with blanks so the
    app degrades gracefully rather than crashing on a slightly different
    template.
    """
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
    return _normalize_columns(raw)


def dummy_schema() -> pd.DataFrame:
    """A small generic e-commerce-style schema used when the user provides none."""
    rows = [
        ("customers", "customer_id", "NUMBER(10)", "N", "Y", ""),
        ("customers", "full_name", "VARCHAR2(120)", "N", "N", ""),
        ("customers", "email", "VARCHAR2(160)", "N", "N", ""),
        ("customers", "created_at", "DATE", "N", "N", ""),
        ("orders", "order_id", "NUMBER(10)", "N", "Y", ""),
        ("orders", "customer_id", "NUMBER(10)", "N", "N", "customers.customer_id"),
        ("orders", "order_status", "VARCHAR2(20)", "N", "N", ""),
        ("orders", "total_amount", "NUMBER(12,2)", "N", "N", ""),
        ("orders", "order_date", "DATE", "N", "N", ""),
        ("order_items", "order_item_id", "NUMBER(10)", "N", "Y", ""),
        ("order_items", "order_id", "NUMBER(10)", "N", "N", "orders.order_id"),
        ("order_items", "product_id", "NUMBER(10)", "N", "N", "products.product_id"),
        ("order_items", "quantity", "NUMBER(6)", "N", "N", ""),
        ("order_items", "unit_price", "NUMBER(12,2)", "N", "N", ""),
        ("products", "product_id", "NUMBER(10)", "N", "Y", ""),
        ("products", "product_name", "VARCHAR2(160)", "N", "N", ""),
        ("products", "stock_quantity", "NUMBER(10)", "N", "N", ""),
        ("products", "unit_price", "NUMBER(12,2)", "N", "N", ""),
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def schema_to_prompt_text(df: pd.DataFrame) -> str:
    """Render a schema DataFrame as a compact DDL-like text block for the LLM prompt."""
    if df is None or df.empty:
        return ""
    lines = []
    for table, group in df.groupby("table_name", sort=False):
        if not str(table).strip():
            continue
        lines.append(f"TABLE {table} (")
        for _, row in group.iterrows():
            bits = [f"  {row['column_name']} {row['data_type']}".rstrip()]
            if _flag(row.get("primary_key")) is True:
                bits.append("PRIMARY KEY")
            # Only assert NOT NULL when the sheet actually says so. A blank
            # cell means "unknown" -- inventing a constraint here would feed
            # the model a schema fact the user never supplied.
            if _flag(row.get("nullable")) is False:
                bits.append("NOT NULL")
            fk = str(row.get("foreign_key", "") or "").strip()
            if fk and fk.lower() != "nan":
                bits.append(f"REFERENCES {fk}")
            lines.append(" ".join(bits))
        lines.append(")")
    return "\n".join(lines)


def _flag(value) -> bool | None:
    """Interpret a spreadsheet yes/no cell. Returns None when unknown/blank."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"y", "yes", "true", "1", "t"}:
        return True
    if text in {"n", "no", "false", "0", "f"}:
        return False
    return None


# ---------------------------------------------------------------------------
# Future part (per PRD section "Future part"): live DB connection to pull
# schema directly instead of an Excel upload. Not implemented yet -- this is
# a placeholder so the UI can show a disabled "Connect to database" option
# without the rest of the app needing to change when it lands.
# ---------------------------------------------------------------------------


def connect_to_database(connection_string: str) -> pd.DataFrame:  # pragma: no cover
    """Placeholder for a future live-DB schema introspection feature.

    Not implemented in this version of the app. Raises NotImplementedError
    so callers fail loudly instead of silently returning nothing.
    """
    raise NotImplementedError(
        "Live database connections are a planned future capability and are "
        "not implemented yet. Please use the Excel upload or the bundled "
        "dummy schema for now."
    )
