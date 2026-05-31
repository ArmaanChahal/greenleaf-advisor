# GreenLeaf Advisor Recommendations

## 1. Plot Priority

The highest-priority plot is **P0067** with an urgency score of **150.826**.

This plot should be reviewed first because its score combines:
- total alerts
- average plant stress
- response delay
- unresolved alert cases

## 2. Response Timing

Response timing results should be interpreted as **association, not proven causality**.

Negative stress delta means plant stress improved. Positive stress delta means plant stress worsened.

The advisor should prioritize alert types and delay buckets that show slower response timing or weaker stress recovery.

## 3. Precision ROI

Precision actions accounted for **4.6%** of total input cost.

Total precision benefit was **$47,034**.

This suggests that precision-supported decisions can be discussed as a business value driver when paired with verified cost, ROI, and season outcome data.

## 4. Financing Summary

Across season outcomes:
- Average season ROI: **42.8%**
- Total season profit: **$71,654**
- Total precision benefit: **$47,034**
- Average yield: **6.6 kg/m2**
- Average marketable ratio: **90.8%**

These verified metrics can support a lender-ready financing memo because they connect cost, profit, ROI, yield, marketability, and precision benefit.

## Advisor Guardrails

The advisor must:
- use only verified Tableau/Python-generated KPI values
- avoid inventing unsupported numbers
- describe response timing as association, not proven causality
- return structured JSON before plain-English recommendations
- explain every recommendation using available KPI evidence
