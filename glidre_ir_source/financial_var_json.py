import json
from pathlib import Path
from datetime import datetime

OUTPUT_FILE = Path("ai_financial_variables_taxonomy.json")

financial_taxonomy = {
    "metadata": {
        "title": "AI Infrastructure Contract and Accounting Variable Taxonomy",
        "description": "Classified JSON structure for accounting, finance, risk, contract, and infrastructure variables discussed in the OpenAI/AWS/Microsoft/NVIDIA comparison.",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": "1.0"
    },

    "categories": {
        "accounting_variables": {
            "assets": [
                "Cash",
                "Accounts Receivable",
                "Investment in OpenAI",
                "Loan Receivable",
                "Computer Equipment",
                "GPU Assets",
                "Data Centers",
                "Property, Plant and Equipment",
                "Cloud Infrastructure",
                "Inventory",
                "Prepaid Assets",
                "Goodwill",
                "Intangible Assets"
            ],
            "liabilities": [
                "Accounts Payable",
                "Loan Payable",
                "Contract Liability",
                "Deferred Revenue",
                "Guarantee Liability",
                "Purchase Commitments",
                "Long-term Debt",
                "Lease Obligations"
            ],
            "equity": [
                "Common Stock",
                "Preferred Stock",
                "Shareholders Equity",
                "Additional Paid-In Capital",
                "Retained Earnings",
                "Equity Investment"
            ],
            "income_statement": [
                "Revenue",
                "Cloud Revenue",
                "Hardware Revenue",
                "Operating Expense",
                "Cloud Computing Expense",
                "Cost of Goods Sold",
                "Gross Profit",
                "Operating Income",
                "Net Income",
                "Depreciation Expense",
                "Impairment Loss",
                "Credit Loss Expense",
                "Bad Debt Expense"
            ],
            "cash_flow": [
                "Operating Cash Flow",
                "Investing Cash Flow",
                "Financing Cash Flow",
                "Capital Expenditures",
                "Free Cash Flow"
            ]
        },

        "investment_variables": [
            "Equity Investment",
            "Preferred Equity",
            "Venture Investment",
            "Fair Value",
            "Carrying Value",
            "Book Value",
            "Market Value",
            "Unrealized Gain",
            "Unrealized Loss",
            "Realized Gain",
            "Investment Return"
        ],

        "credit_risk_variables": [
            "Credit Risk",
            "Counterparty Risk",
            "Probability of Default",
            "Expected Credit Loss",
            "Allowance for Credit Losses",
            "Bad Debt",
            "Loan Default",
            "Recoverability",
            "Credit Exposure"
        ],

        "contract_variables": [
            "Purchase Agreement",
            "Cloud Service Contract",
            "Long-term Commitment",
            "Purchase Commitment",
            "Minimum Purchase Requirement",
            "Cancellation Clause",
            "Contract Duration",
            "Contract Value",
            "Service-Level Agreement",
            "Pricing Terms",
            "Fair Market Pricing",
            "Arm's-Length Transaction"
        ],

        "revenue_recognition_variables": [
            "Revenue Recognition",
            "Performance Obligation",
            "Delivery of Service",
            "Deferred Revenue",
            "Contract Asset",
            "Contract Liability",
            "Collectability",
            "Economic Substance"
        ],

        "valuation_variables": [
            "Fair Value",
            "Discounted Cash Flow",
            "Net Present Value",
            "Internal Rate of Return",
            "Enterprise Value",
            "Market Capitalization",
            "Asset Impairment",
            "Recoverable Amount"
        ],

        "infrastructure_variables": [
            "GPU Purchases",
            "Cloud Capacity",
            "Compute Capacity",
            "Data Center Capacity",
            "Server Deployment",
            "Network Infrastructure",
            "Fiber Optics",
            "Bandwidth",
            "AI Infrastructure"
        ],

        "strategic_business_variables": [
            "Strategic Alliance",
            "Partnership",
            "Joint Investment",
            "Vendor Financing",
            "Capacity Swap",
            "Reciprocal Agreement",
            "Cross-Investment",
            "Supply Agreement",
            "Long-Term Supply Contract",
            "Preferred Supplier",
            "Customer Concentration"
        ],

        "risk_variables": [
            "Liquidity Risk",
            "Solvency Risk",
            "Market Risk",
            "Operational Risk",
            "Concentration Risk",
            "Technology Risk",
            "Execution Risk",
            "Counterparty Exposure",
            "Contract Risk"
        ],

        "financial_ratios": [
            "Return on Assets",
            "Return on Equity",
            "Gross Margin",
            "Operating Margin",
            "Net Margin",
            "Debt-to-Equity",
            "Current Ratio",
            "Quick Ratio",
            "Interest Coverage",
            "Asset Turnover"
        ],

        "ai_industry_variables": [
            "GPU Procurement",
            "AI Compute",
            "Model Training Cost",
            "Inference Cost",
            "Token Cost",
            "Cloud Utilization",
            "Compute Demand",
            "Model Scaling",
            "Capital Intensity",
            "Infrastructure Utilization"
        ],

        "historical_telecom_variables": [
            "Fiber Capacity",
            "Indefeasible Right of Use",
            "Capacity Swap",
            "Vendor Financing",
            "Network Build-Out",
            "Dark Fiber",
            "Bandwidth Sales",
            "Telecommunications Equipment",
            "Infrastructure Expansion"
        ],

        "economic_variables": [
            "Capital Allocation",
            "Investment Cycle",
            "Cash Burn",
            "Profitability",
            "Revenue Growth",
            "Earnings Quality",
            "Capital Formation",
            "Return on Investment",
            "Productivity",
            "Economic Value Creation"
        ],

        "corporate_governance_variables": [
            "Material Disclosure",
            "Related-Party Transactions",
            "Audit Committee Oversight",
            "Financial Statement Disclosure",
            "SEC Reporting",
            "Accounting Policy",
            "Risk Factors"
        ]
    }
}


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def print_summary(data):
    print("=" * 80)
    print(data["metadata"]["title"])
    print("=" * 80)

    for category_name, category_content in data["categories"].items():
        print(f"\nCATEGORY: {category_name}")

        if isinstance(category_content, dict):
            for subcategory, variables in category_content.items():
                print(f"  {subcategory}: {len(variables)} variables")
        elif isinstance(category_content, list):
            print(f"  {len(category_content)} variables")


def main():
    save_json(financial_taxonomy, OUTPUT_FILE)
    print_summary(financial_taxonomy)
    print(f"\nJSON file saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()