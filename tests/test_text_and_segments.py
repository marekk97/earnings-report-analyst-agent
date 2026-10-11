import pytest

from earnings_analyst.provenance import DataError
from earnings_analyst.sec.filings import extract_section, focus_text, guidance_excerpts, html_to_text, normalize_section
from earnings_analyst.sec.segments import parse_segments, summarise

FILLER = "Revenue increased due to higher pricing and strong demand in North America. " * 12

TEN_Q_HTML = f"""
<html><body>
<p>Table of Contents</p>
<table><tr><td>Item 2. Management's Discussion and Analysis</td><td>15</td></tr>
<tr><td>Item 3. Quantitative and Qualitative Disclosures About Market Risk</td><td>30</td></tr></table>
<p>PART I - FINANCIAL INFORMATION</p>
<p>Item 1. Financial Statements</p><p>Numbers...</p>
<p>ITEM 2. MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
<p>Overview</p><p>{FILLER}</p>
<p>Gross margin increased 60 basis points to 42.8% due to lower logistics costs.</p>
<p>{FILLER}</p>
<p>ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</p><p>Rates...</p>
</body></html>
"""


def test_html_to_text_keeps_paragraphs_and_tables():
    text = html_to_text("<div><p>Hello&nbsp;world</p><table><tr><td>Revenue</td><td>$11.2</td></tr></table></div>")
    assert "Hello world" in text and "Revenue | $11.2" in text


def test_extract_section_skips_table_of_contents():
    section = extract_section(html_to_text(TEN_Q_HTML), "10-Q", "mdna")
    assert section.startswith("ITEM 2")
    assert "42.8%" in section and "Rates" not in section


def test_missing_section_is_structured():
    with pytest.raises(DataError) as exc:
        extract_section("nothing here", "10-K", "mdna")
    assert exc.value.code == "section_not_found"
    with pytest.raises(DataError):
        extract_section("x", "10-Q", "business")


def test_section_aliases():
    assert normalize_section("MD&A") == "mdna"
    assert normalize_section("Risk Factors") == "risk_factors"


def test_focus_text_selects_relevant_paragraphs():
    text = "\n".join(["Intro paragraph about the company."] * 20 + ["Gross margin rose on pricing."] + ["Other."] * 20)
    chunk, truncated = focus_text(text, "gross margin", 200)
    assert "Gross margin rose" in chunk and truncated
    page, more = focus_text("abcdef", None, 4)
    assert page == "abcd" and more


def test_guidance_excerpts_skip_safe_harbor():
    text = ("This release contains forward-looking statements; we expect many risks.\n"
            "For fiscal 2027, the company expects revenue to decline high-single digits.\n"
            "Revenue was $11.2 billion.")
    out = guidance_excerpts(text)
    assert out == ["For fiscal 2027, the company expects revenue to decline high-single digits."]


INSTANCE = """<?xml version="1.0"?>
<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
      xmlns:us-gaap="http://fasb.org/us-gaap/2025" xmlns:acme="http://acme.com/2025">
 <context id="c1"><entity><identifier scheme="x">1</identifier><segment>
   <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">acme:WidgetsSegmentMember</xbrldi:explicitMember>
   <xbrldi:explicitMember dimension="srt:ConsolidationItemsAxis">us-gaap:OperatingSegmentsMember</xbrldi:explicitMember>
 </segment></entity><period><startDate>2025-07-01</startDate><endDate>2025-09-30</endDate></period></context>
 <context id="c2"><entity><identifier scheme="x">1</identifier><segment>
   <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">acme:GadgetsSegmentMember</xbrldi:explicitMember>
 </segment></entity><period><startDate>2025-07-01</startDate><endDate>2025-09-30</endDate></period></context>
 <context id="c3"><entity><identifier scheme="x">1</identifier><segment>
   <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">acme:WidgetsSegmentMember</xbrldi:explicitMember>
   <xbrldi:explicitMember dimension="us-gaap:ContractWithCustomerSalesChannelAxis">acme:OnlineMember</xbrldi:explicitMember>
 </segment></entity><period><startDate>2025-07-01</startDate><endDate>2025-09-30</endDate></period></context>
 <context id="c4"><entity><identifier scheme="x">1</identifier><segment>
   <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">acme:WidgetsSegmentMember</xbrldi:explicitMember>
 </segment></entity><period><startDate>2025-01-01</startDate><endDate>2025-09-30</endDate></period></context>
 <us-gaap:Revenues contextRef="c1" unitRef="usd" decimals="-6">80000000</us-gaap:Revenues>
 <acme:SegmentEarningsBeforeInterestAndTaxes contextRef="c1" unitRef="usd" decimals="-6">20000000</acme:SegmentEarningsBeforeInterestAndTaxes>
 <us-gaap:Revenues contextRef="c2" unitRef="usd" decimals="-6">52000000</us-gaap:Revenues>
 <us-gaap:OperatingIncomeLoss contextRef="c2" unitRef="usd" decimals="-6">-3000000</us-gaap:OperatingIncomeLoss>
 <us-gaap:Revenues contextRef="c3" unitRef="usd" decimals="-6">30000000</us-gaap:Revenues>
 <us-gaap:Revenues contextRef="c4" unitRef="usd" decimals="-6">240000000</us-gaap:Revenues>
</xbrl>"""


def test_parse_segments_accepts_consolidation_axis_and_skips_sub_breakdowns():
    parsed = parse_segments(INSTANCE, "segment", "2025-09-30")
    rows = summarise(parsed)
    quarter = next(r for r in rows if r["start"] == "2025-07-01")
    by = {m["member"]: m for m in quarter["members"]}
    assert by["Widgets"]["revenue"] == 80e6  # the sales-channel sub-breakdown (30M) is excluded
    assert by["Widgets"]["operating_income"] == 20e6 and by["Widgets"]["profit_concept"].startswith("acme:")
    assert by["Gadgets"]["operating_income"] == -3e6
    assert any(r["start"] == "2025-01-01" for r in rows)  # YTD kept separately


def test_parse_segments_missing_dimension():
    with pytest.raises(DataError) as exc:
        parse_segments(INSTANCE, "geography", "2025-09-30")
    assert exc.value.code == "segments_not_reported"
