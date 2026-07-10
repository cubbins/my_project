"""
SEC AI Infrastructure Research Agent

Questions:
1. Which companies have reciprocal investment and supplier relationships?
2. Which AI companies have greatest customer concentration risk?
3. How do cloud purchase commitments correlate with capex over time?
4. Which partnerships expanded/contracted after profitability or liquidity changes?

Install:
    pip install requests

Run:
    python sec_ai_research_agent.py

Important:
    Edit USER_AGENT before running.
"""

import json
import re
import time
from pathlib import Path
from datetime import datetime
from html import unescape

import requests


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

USER_AGENT = "Thomas Cubbins research contact@example.com"

OUTPUT_DIR = Path("sec_ai_research_output")
OUTPUT_DIR.mkdir(exist_ok=True)

COMPANIES = {
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "ORCL": "Oracle",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "INTC": "Intel"
}

FORMS_TO_SCAN = {"10-K", "10-Q", "8-K"}

KEYWORD_GROUPS = {
    "reciprocal_investment_supplier": [
        "strategic investment",
        "commercial agreement",
        "supply agreement",
        "purchase commitment",
        "cloud services",
        "compute",
        "artificial intelligence",
        "AI infrastructure",
        "customer",
        "supplier",
        "partner",
        "equity investment",
    ],
    "customer_concentration": [
        "customer concentration",
        "significant customer",
        "major customer",
        "one customer",
        "customers accounted for",
        "dependence on",
        "concentration of revenue",
    ],
    "cloud_purchase_commitments": [
        "purchase obligations",
        "cloud",
        "data center",
        "compute capacity",
        "minimum purchase",
        "contractual commitments",
        "remaining performance obligations",
    ],
    "profitability_liquidity": [
        "net income",
        "operating income",
        "liquidity",
        "cash flows",
        "capital expenditures",
        "free cash flow",
        "cash and cash equivalents",
        "debt",
    ],
}


# ---------------------------------------------------------------------
# SEC CLIENT
# ---------------------------------------------------------------------

class SECClient:
    def __init__(self, user_agent):
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }

    def get_json(self, url):
        time.sleep(0.2)
        r = requests.get(url, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_text(self, url):
        headers = dict(self.headers)
        headers["Host"] = "www.sec.gov"
        time.sleep(0.2)
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.text

    def get_company_tickers(self):
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def ticker_to_cik(self, ticker):
        data = self.get_company_tickers()
        ticker = ticker.upper()

        for _, row in data.items():
            if row["ticker"].upper() == ticker:
                return str(row["cik_str"]).zfill(10)

        raise ValueError(f"Ticker not found: {ticker}")

    def submissions(self, cik):
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        return self.get_json(url)

    def companyfacts(self, cik):
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        return self.get_json(url)


# ---------------------------------------------------------------------
# FINANCIAL FACT EXTRACTION
# ---------------------------------------------------------------------

def extract_us_gaap_series(companyfacts, concept_names):
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    output = {}

    for concept in concept_names:
        if concept not in facts:
            continue

        units = facts[concept].get("units", {})
        usd_items = units.get("USD", [])

        yearly = []
        for item in usd_items:
            form = item.get("form")
            fy = item.get("fy")
            fp = item.get("fp")
            val = item.get("val")
            filed = item.get("filed")

            if form in {"10-K", "10-Q"} and fy and val is not None:
                yearly.append({
                    "concept": concept,
                    "fy": fy,
                    "fp": fp,
                    "form": form,
                    "value": val,
                    "filed": filed
                })

        output[concept] = yearly

    return output


def latest_annual_value(series, concept):
    items = [
        x for x in series.get(concept, [])
        if x.get("form") == "10-K" and x.get("fp") == "FY"
    ]

    if not items:
        return None

    items = sorted(items, key=lambda x: (x["fy"], x.get("filed", "")))
    return items[-1]["value"]


def compute_financial_variables(series):
    concepts = {
        "Revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
        "NetIncome": ["NetIncomeLoss"],
        "OperatingIncome": ["OperatingIncomeLoss"],
        "Assets": ["Assets"],
        "Liabilities": ["Liabilities"],
        "Cash": ["CashAndCashEquivalentsAtCarryingValue"],
        "CapEx": ["PaymentsToAcquirePropertyPlantAndEquipment"],
        "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    }

    result = {}

    for label, candidates in concepts.items():
        for concept in candidates:
            val = latest_annual_value(series, concept)
            if val is not None:
                result[label] = val
                break

    revenue = result.get("Revenue")
    net_income = result.get("NetIncome")
    assets = result.get("Assets")
    liabilities = result.get("Liabilities")
    capex = result.get("CapEx")
    ocf = result.get("OperatingCashFlow")

    ratios = {}

    if revenue and net_income is not None:
        ratios["NetMargin"] = net_income / revenue

    if assets and liabilities is not None:
        ratios["LiabilitiesToAssets"] = liabilities / assets

    if ocf is not None and capex is not None:
        ratios["ApproxFreeCashFlow"] = ocf - abs(capex)

    result["ratios"] = ratios
    return result


# ---------------------------------------------------------------------
# FILING DOWNLOAD / TEXT SCAN
# ---------------------------------------------------------------------

def clean_html_text(raw):
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = unescape(raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def get_recent_filings(submissions, max_filings=5):
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    filings = []

    for form, accession, doc, filing_date in zip(
        forms, accession_numbers, primary_docs, filing_dates
    ):
        if form in FORMS_TO_SCAN:
            filings.append({
                "form": form,
                "accession": accession,
                "primary_doc": doc,
                "filing_date": filing_date
            })

        if len(filings) >= max_filings:
            break

    return filings


def filing_url(cik, accession, primary_doc):
    cik_no_zero = str(int(cik))
    accession_clean = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zero}/{accession_clean}/{primary_doc}"
    )


def keyword_scan(text, keyword_groups):
    findings = {}

    lower_text = text.lower()

    for group, keywords in keyword_groups.items():
        hits = []

        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in lower_text:
                idx = lower_text.find(kw_lower)
                start = max(0, idx - 250)
                end = min(len(text), idx + 500)

                hits.append({
                    "keyword": kw,
                    "context": text[start:end]
                })

        findings[group] = hits

    return findings


# ---------------------------------------------------------------------
# RESEARCH AGENT
# ---------------------------------------------------------------------

def analyze_company(sec, ticker, company_name):
    cik = sec.ticker_to_cik(ticker)

    print(f"Analyzing {ticker} / {company_name} / CIK {cik}")

    submissions = sec.submissions(cik)
    facts = sec.companyfacts(cik)

    financial_concepts = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "NetIncomeLoss",
        "OperatingIncomeLoss",
        "Assets",
        "Liabilities",
        "CashAndCashEquivalentsAtCarryingValue",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "NetCashProvidedByUsedInOperatingActivities",
    ]

    series = extract_us_gaap_series(facts, financial_concepts)
    financial_variables = compute_financial_variables(series)

    filings = get_recent_filings(submissions, max_filings=5)

    filing_results = []

    for filing in filings:
        url = filing_url(cik, filing["accession"], filing["primary_doc"])

        try:
            raw = sec.get_text(url)
            text = clean_html_text(raw)
            scan = keyword_scan(text, KEYWORD_GROUPS)

            filing_results.append({
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "url": url,
                "keyword_findings": scan
            })

        except Exception as exc:
            filing_results.append({
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "url": url,
                "error": str(exc)
            })

    return {
        "ticker": ticker,
        "company_name": company_name,
        "cik": cik,
        "financial_variables": financial_variables,
        "xbrl_series": series,
        "filing_scans": filing_results
    }


def score_company(company_result):
    score = {
        "reciprocal_relationship_score": 0,
        "customer_concentration_score": 0,
        "cloud_commitment_score": 0,
        "profitability_liquidity_score": 0
    }

    group_to_score = {
        "reciprocal_investment_supplier": "reciprocal_relationship_score",
        "customer_concentration": "customer_concentration_score",
        "cloud_purchase_commitments": "cloud_commitment_score",
        "profitability_liquidity": "profitability_liquidity_score"
    }

    for filing in company_result.get("filing_scans", []):
        findings = filing.get("keyword_findings", {})

        for group, hits in findings.items():
            score_key = group_to_score.get(group)
            if score_key:
                score[score_key] += len(hits)

    company_result["scores"] = score
    return company_result


# ---------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------

def write_json_report(results):
    output = {
        "metadata": {
            "title": "SEC AI Infrastructure Relationship Research Agent Output",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "questions": [
                "Which companies have reciprocal investment and supplier relationships?",
                "Which AI companies have the greatest customer concentration risk?",
                "How do cloud purchase commitments correlate with capital expenditures over time?",
                "Which strategic partnerships expanded or contracted following changes in profitability or liquidity?"
            ],
            "source": "SEC EDGAR submissions and XBRL companyfacts APIs"
        },
        "results": results
    }

    path = OUTPUT_DIR / "sec_ai_research_results.json"
    path.write_text(json.dumps(output, indent=4), encoding="utf-8")
    return path


def write_markdown_report(results):
    lines = []

    lines.append("# SEC AI Infrastructure Research Agent Report\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")

    lines.append("## Research Questions\n")
    lines.append("1. Which companies have reciprocal investment and supplier relationships?")
    lines.append("2. Which AI companies have the greatest customer concentration risk?")
    lines.append("3. How do cloud purchase commitments correlate with capital expenditures over time?")
    lines.append("4. Which strategic partnerships expanded or contracted following profitability or liquidity changes?\n")

    lines.append("## Company Scores\n")
    lines.append("| Company | Ticker | Reciprocal Relationship | Customer Concentration | Cloud Commitment | Profitability/Liquidity |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for r in results:
        s = r["scores"]
        lines.append(
            f"| {r['company_name']} | {r['ticker']} | "
            f"{s['reciprocal_relationship_score']} | "
            f"{s['customer_concentration_score']} | "
            f"{s['cloud_commitment_score']} | "
            f"{s['profitability_liquidity_score']} |"
        )

    lines.append("\n## Latest Financial Variables\n")

    for r in results:
        fv = r["financial_variables"]
        lines.append(f"### {r['company_name']} ({r['ticker']})\n")
        lines.append("```json")
        lines.append(json.dumps(fv, indent=4))
        lines.append("```\n")

    lines.append("## Extracted Filing Evidence\n")

    for r in results:
        lines.append(f"### {r['company_name']} ({r['ticker']})\n")

        for filing in r.get("filing_scans", []):
            lines.append(f"#### {filing.get('form')} filed {filing.get('filing_date')}")
            lines.append(f"Source: {filing.get('url')}\n")

            findings = filing.get("keyword_findings", {})

            for group, hits in findings.items():
                if not hits:
                    continue

                lines.append(f"**{group}**\n")

                for hit in hits[:3]:
                    context = hit["context"].replace("|", " ")
                    lines.append(f"- Keyword: `{hit['keyword']}`")
                    lines.append(f"  - Context: {context[:700]}...\n")

    path = OUTPUT_DIR / "sec_ai_research_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    sec = SECClient(USER_AGENT)

    results = []

    for ticker, company_name in COMPANIES.items():
        try:
            result = analyze_company(sec, ticker, company_name)
            result = score_company(result)
            results.append(result)

        except Exception as exc:
            results.append({
                "ticker": ticker,
                "company_name": company_name,
                "error": str(exc)
            })

    json_path = write_json_report(results)
    md_path = write_markdown_report(results)

    print("\nDone.")
    print(f"JSON report: {json_path.resolve()}")
    print(f"Markdown report: {md_path.resolve()}")


if __name__ == "__main__":
    main()