"""Segment / geography / product breakouts from a filing's XBRL instance.

The companyfacts API drops dimensional facts, so segment data comes from the instance document of a
specific 10-Q/10-K. Since ASU 2023-07, segment facts are often tagged with two dimensions: the segment
axis plus `srt:ConsolidationItemsAxis = OperatingSegmentsMember`. That pairing is accepted; any other
extra dimension (sales channel, product line inside a segment, ...) is a sub-breakdown and is skipped.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from lxml import etree

from ..provenance import DataError

XBRLI = "{http://www.xbrl.org/2003/instance}"
XBRLDI = "{http://xbrl.org/2006/xbrldi}"

AXES = {
    "segment": ["us-gaap:StatementBusinessSegmentsAxis"],
    # Business segments count as geography when the members are regions (Apple: Americas, Europe, ...).
    "geography": ["srt:StatementGeographicalAxis", "us-gaap:SubsegmentsAxis", "us-gaap:StatementBusinessSegmentsAxis"],
    "product": ["srt:ProductOrServiceAxis"],
}
NEUTRAL_MEMBERS = {("srt:ConsolidationItemsAxis", "us-gaap:OperatingSegmentsMember")}

REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
PROFIT_CONCEPTS = [
    "OperatingIncomeLoss",
    "SegmentReportingInformationOperatingIncomeLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "GrossProfit",
]
GEO_WORDS = re.compile(
    r"america|europe|emea|asia|pacific|china|japan|africa|middle ?east|latin|international|domestic|"
    r"\bus\b|united ?states|canada|mexico|india|rest ?of|other ?countries|foreign|region",
    re.I,
)
EXTENSION_PROFIT_RE = re.compile(r"OperatingIncome|OperatingProfit|SegmentProfit|EarningsBeforeInterestAndTaxes|SegmentIncome|SegmentEBIT", re.I)


@dataclass
class _Ctx:
    members: dict[str, str]
    start: str | None
    end: str


def _humanize(member: str) -> str:
    local = member.split(":")[-1]
    local = re.sub(r"(Segment)?Member$", "", local)
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", local).strip()
    return re.sub(r"^I (?=(Phone|Pad|Pod|Mac|Cloud|Tunes)\b)", "i", spaced)  # "I Phone" -> "iPhone"


def _flag_subtotals(items: list[dict]) -> None:
    """Mark members whose revenue equals the sum of two or more other members (e.g. Apple's "Products" line)."""
    vals = [(i, m["revenue"]) for i, m in enumerate(items) if m["revenue"]]
    if len(vals) > 14:
        return
    for i, total in vals:
        others = [(j, v) for j, v in vals if j != i and v < total]
        for mask in range(1, 1 << len(others)):
            picked = [v for k, (_, v) in enumerate(others) if mask >> k & 1]
            if len(picked) >= 2 and abs(sum(picked) - total) <= 0.001 * total:
                items[i]["is_subtotal"] = True
                break


def parse_segments(xml: str | bytes, dimension: str, period_end: str) -> dict:
    """Revenue and profit by member for every duration ending on `period_end` in the instance."""
    if dimension not in AXES:
        raise DataError("bad_dimension", f"dimension must be one of {sorted(AXES)}")
    root = etree.fromstring(xml.encode() if isinstance(xml, str) else xml, parser=etree.XMLParser(huge_tree=True))
    ctxs: dict[str, _Ctx] = {}
    for c in root.iter(f"{XBRLI}context"):
        p = c.find(f"{XBRLI}period")
        members = {m.get("dimension"): (m.text or "").strip() for m in c.iter(f"{XBRLDI}explicitMember")}
        ctxs[c.get("id")] = _Ctx(members, p.findtext(f"{XBRLI}startDate"), p.findtext(f"{XBRLI}endDate") or p.findtext(f"{XBRLI}instant"))

    best: dict | None = None
    for axis in AXES[dimension]:
        rows: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)  # (start, end) -> member -> {concept: value}
        for el in root:
            ref = el.get("contextRef")
            ctx = ctxs.get(ref) if ref else None
            if not ctx or not ctx.start or ctx.end != period_end or axis not in ctx.members:
                continue
            others = {(d, m) for d, m in ctx.members.items() if d != axis}
            # Subsegments are nested under a business segment; that parent dimension is expected.
            others = {o for o in others if not (axis == "us-gaap:SubsegmentsAxis" and o[0] == "us-gaap:StatementBusinessSegmentsAxis")}
            if not others <= NEUTRAL_MEMBERS:
                continue
            qn = etree.QName(el)
            prefix = el.prefix or ""
            concept = qn.localname
            if el.text is None or not re.fullmatch(r"-?[\d.]+(E-?\d+)?", el.text.strip()):
                continue
            is_revenue = prefix == "us-gaap" and concept in REVENUE_CONCEPTS
            is_profit = (prefix == "us-gaap" and concept in PROFIT_CONCEPTS) or (
                prefix != "us-gaap" and EXTENSION_PROFIT_RE.search(concept)
            )
            if not (is_revenue or is_profit):
                continue
            entry = rows[(ctx.start, ctx.end)].setdefault(ctx.members[axis], {})
            entry[f"{prefix}:{concept}"] = float(el.text)
        # Several axes can carry data (Nike tags a lone "US" figure on the geographic axis and its four sales
        # regions on the subsegment axis): keep the one with the most members.
        if dimension == "geography" and axis == "us-gaap:StatementBusinessSegmentsAxis":
            names = {_humanize(m) for members in rows.values() for m in members}
            if not names or sum(bool(GEO_WORDS.search(n)) for n in names) < 0.75 * len(names):
                continue  # product/business segments, not regions
        width = max((len(m) for m in rows.values()), default=0)
        if width and (best is None or width > best["width"]):
            best = {"axis": axis, "periods": rows, "width": width}
    if best:
        return {"axis": best["axis"], "periods": best["periods"]}
    raise DataError(
        "segments_not_reported",
        f"No {dimension} breakdown tagged in this filing's XBRL for the period ending {period_end}.",
        tried_axes=AXES[dimension],
    )


def summarise(parsed: dict) -> list[dict]:
    """Collapse parsed rows into a list of periods with one revenue and one profit figure per member."""
    out = []
    for (start, end), members in sorted(parsed["periods"].items(), key=lambda kv: kv[0], reverse=True):
        items = []
        for member, concepts in members.items():
            rev_key = next((f"us-gaap:{c}" for c in REVENUE_CONCEPTS if f"us-gaap:{c}" in concepts), None)
            profit_key = next((f"us-gaap:{c}" for c in PROFIT_CONCEPTS if f"us-gaap:{c}" in concepts), None)
            if profit_key is None:
                profit_key = next((k for k in concepts if not k.startswith("us-gaap:")), None)
            items.append(
                {
                    "member": _humanize(member),
                    "xbrl_member": member,
                    "revenue": concepts.get(rev_key) if rev_key else None,
                    "revenue_concept": rev_key,
                    "operating_income": concepts.get(profit_key) if profit_key else None,
                    "profit_concept": profit_key,
                }
            )
        items.sort(key=lambda r: -(r["revenue"] or 0))
        _flag_subtotals(items)
        out.append({"start": start, "end": end, "members": items})
    return out
