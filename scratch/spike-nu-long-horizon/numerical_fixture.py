"""
Spike ν — production-scale numerical fixture (Surface 2).

A synthetic report with 100+ distinct figures across markdown tables,
percentages, currency, dates, and inline statistics. Fed to the
response-synthesizer to test numerical fidelity at production scale
(extends Spike ε' B1's 25-figure fixture).

All figures are synthetic. Distinctness is deliberate: repeated values
would blur the number-overlap fidelity analysis.
"""

from __future__ import annotations

NUMERICAL_FIXTURE = """\
GLOBAL CLOUD INFRASTRUCTURE MARKET REPORT — SYNTHETIC FIXTURE (Spike ν Surface 2)

Section 1 — Provider revenue and headcount (FY2024)

| Provider     | Revenue (USD M) | Employees | Founded | Market cap (USD B) | Data centers |
|--------------|-----------------|-----------|---------|--------------------|--------------|
| Aurora Cloud | 48,217          | 91,430    | 2006    | 1,284              | 137          |
| Beacon Sys   | 33,905          | 64,218    | 2009    | 902                | 88           |
| Cirrus Net   | 27,461          | 52,007    | 2011    | 671                | 74           |
| Delta Grid   | 19,338          | 38,915    | 2013    | 443                | 61           |
| Equinox Edge | 12,774          | 24,602    | 2015    | 318                | 49           |
| Forge Data   | 8,651           | 17,330    | 2017    | 196                | 33           |

Section 2 — Regional revenue split (FY2024, USD millions)

- North America: 71,442 (up from 63,118 in FY2023)
- Europe: 38,907 (up from 34,225 in FY2023)
- Asia-Pacific: 29,613 (up from 22,884 in FY2023)
- Latin America: 7,206 (up from 5,941 in FY2023)
- Middle East & Africa: 3,628 (up from 2,755 in FY2023)

Section 3 — Year-over-year growth rates (%)

Aurora Cloud 18.3%, Beacon Sys 14.7%, Cirrus Net 22.1%, Delta Grid 9.6%,
Equinox Edge 31.4%, Forge Data 44.2%. Blended market growth: 19.8%.

Section 4 — Service-line margins (gross margin %, FY2024)

Compute 62.4, Storage 58.1, Networking 49.7, Managed databases 67.3,
AI inference 71.9, Security services 55.2.

Section 5 — Operational metrics

Aggregate uptime across providers averaged 99.982% in 2024. Mean latency
to nearest edge node was 11.4 ms in North America, 14.8 ms in Europe,
22.6 ms in Asia-Pacific. Total power consumption reached 28.7 TWh, of
which 63.5% was sourced from renewables. Average PUE (power usage
effectiveness) improved to 1.21 from 1.34 the prior year.

Section 6 — Pricing (USD per unit, on-demand, FY2024)

Compute (per vCPU-hour): 0.0412. Block storage (per GB-month): 0.083.
Egress (per GB): 0.087. AI inference (per 1,000 tokens): 0.0024.
Managed Postgres (per instance-hour): 0.146.

Section 7 — Key dates

Aurora Cloud's flagship region us-aurora-1 launched 2006-08-24. The
2024 outage that affected Beacon Sys lasted from 2024-03-11 09:42 UTC to
2024-03-11 13:17 UTC. Cirrus Net's IPO priced at 38 USD per share on
2011-05-19. The market report covers the period 2023-01-01 through
2024-12-31.

Section 8 — Misc statistics

The six providers operated a combined 442 data centers across 61
countries, serving an estimated 4,380,000 business customers. The largest
single facility holds 184,000 servers. Median contract value rose to
217,500 USD, a 12.8% increase. Customer churn held at 4.3%. The sector
added 91,200 net new jobs in 2024.

Section 9 — Quarterly revenue by provider (FY2024, USD millions)

| Provider     | Q1     | Q2     | Q3     | Q4     |
|--------------|--------|--------|--------|--------|
| Aurora Cloud | 11,204 | 11,873 | 12,406 | 12,734 |
| Beacon Sys   | 7,915  | 8,331  | 8,602  | 9,057  |
| Cirrus Net   | 6,108  | 6,742  | 7,019  | 7,592  |
| Delta Grid   | 4,471  | 4,803  | 4,927  | 5,137  |
| Equinox Edge | 2,803  | 3,061  | 3,318  | 3,592  |
| Forge Data   | 1,847  | 2,019  | 2,288  | 2,497  |

Section 10 — R&D and capital expenditure (FY2024, USD millions)

R&D spend: Aurora Cloud 6,742; Beacon Sys 4,118; Cirrus Net 3,205;
Delta Grid 2,114; Equinox Edge 1,538; Forge Data 968. Capital
expenditure: Aurora Cloud 9,361; Beacon Sys 5,827; Cirrus Net 4,402;
Delta Grid 2,956; Equinox Edge 1,874; Forge Data 1,113.

Section 11 — Customer satisfaction and SLAs

Net promoter scores: Aurora Cloud 54, Beacon Sys 47, Cirrus Net 61,
Delta Grid 39, Equinox Edge 66, Forge Data 71. Support ticket
resolution averaged 318 minutes. SLA credits paid totaled 14,907,000 USD
across the year, against contractual targets of 99.95% availability.
"""


# Synthesizer request framings — each invocation tests fidelity under a
# different reproduction demand. Pre-specified before the run.
NUMERICAL_REQUESTS: list[dict] = [
    {
        "id": "N1-summary",
        "request": "Summarize this cloud infrastructure market report, "
        "highlighting the largest provider by revenue, the fastest-growing "
        "provider, and the overall market growth rate.",
    },
    {
        "id": "N2-table-reproduce",
        "request": "Reproduce the provider revenue and headcount table from "
        "this report, and tell me the total revenue across all six providers.",
    },
    {
        "id": "N3-regional-detail",
        "request": "What were the FY2024 and FY2023 regional revenue figures, "
        "and which region grew the most in absolute dollars?",
    },
    {
        "id": "N4-operational",
        "request": "Give me the operational metrics from this report: uptime, "
        "latency by region, power consumption, renewable share, and PUE.",
    },
]
