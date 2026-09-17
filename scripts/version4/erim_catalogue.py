"""ERIM adapter: declared product types parsed from ERIM catalogue labels.

ERIM's product files give a label per UPC but no sub-category. These rules read the product
form each label states (for example oil- or water-packed tuna). They were fixed from the
labels alone, before any co-purchase statistic was computed (paper/ERIM_SUBSTRUCTURE_SCREEN.md).
Categories without a stated form (brownie, ketchup, ddinner), and labels no rule matches,
get no type. This module is ERIM-specific adapter code: the pipeline only sees the resulting
canonical ``subcategory`` column.
"""
from __future__ import annotations

import re


def _sugar(label: str) -> str | None:
    if re.search(r"SGR SUB|SWTNR|EQUAL|SWT LW", label):
        return "sugar substitute"
    if " PW " in f" {label} ":
        return "powdered sugar"
    if "BRN" in label:
        return "brown sugar"
    return "granulated sugar" if "SGR" in label else None


def _margarine(label: str) -> str | None:
    if re.search(r"SQZ|LIQ", label):
        return "squeeze margarine"
    if re.search(r"\b\d?TB\b", label):
        return "tub margarine"
    if re.search(r"\b\d?ST\b", label):
        return "stick margarine"
    return None


def _word(label: str, token: str) -> bool:
    return f" {token} " in f" {label} "


TYPE_RULES = {
    "tuna": lambda s: "oil-packed tuna" if _word(s, "OIL") else ("water-packed tuna" if "WTR" in s else None),
    "pbutter": lambda s: ("creamy peanut butter" if _word(s, "CRM")
                          else "chunky peanut butter" if _word(s, "CHK") else None),
    "sugar": _sugar,
    "marg": _margarine,
    "tissue": lambda s: ("1-ply tissue" if _word(s, "1P")
                         else "2-ply tissue" if _word(s, "2P") else None),
}


def product_type(category: str, label: str) -> str | None:
    rule = TYPE_RULES.get(category)
    return None if rule is None else rule(label)
