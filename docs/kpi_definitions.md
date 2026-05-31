# KPI Definitions

## Total Alerts
Total number of alert rows where Alert Flag equals 1.

Formula:
SUM(Alert Flag)

Purpose:
Shows how many alert conditions were triggered.

## Average Plant Stress
Average Plant Stress Index across selected records.

Formula:
AVG(Plant Stress Index)

Purpose:
Shows overall crop stress level.

## Average Response Delay
Average number of days between an alert and an action.

Formula:
AVG(Action Delay Days)

Purpose:
Shows whether alerts are being handled quickly.

## No Response Count
Number of alert records where no action was taken.

Formula:
IF Alert Flag = 1 AND Action Taken = 0 THEN 1 ELSE 0

Purpose:
Shows unresolved alert cases.

## Urgency Score
Combined score used to rank plots needing attention.

Formula:
AVG(Plant Stress Index) + SUM(Alert Flag) + AVG(Action Delay Days) + No Response Count

Purpose:
Higher score means the plot should be reviewed sooner.

## Response Rate
Percentage of alert rows where action was taken.

Formula:
Alert rows with action taken / total alert rows

Purpose:
Shows how often alerts received a response.

## Same-Day Response Rate
Percentage of alert rows where action was taken on the same day.

Formula:
Alert rows with Action Taken = 1 and Action Delay Days = 0 / total alert rows

Purpose:
Shows how often alerts were handled immediately.

## Average Stress Recovery
Average Post Action Stress Delta 3D.

Formula:
AVG(Post Action Stress Delta 3D)

Purpose:
Negative values mean plant stress improved. Positive values mean plant stress worsened.

## Precision Cost Share
Share of total input cost spent on precision actions.

Formula:
Daily Precision Cost / Daily Total Input Cost

Purpose:
Shows how much of the input budget was targeted precision spending.

## Season ROI
Season-level return on investment by plot.

Purpose:
Shows financial return from the season outcome data.

## Precision Benefit
Estimated dollar benefit connected to precision-supported practices.

Purpose:
Shows business value from precision-supported decisions.

## Marketable Ratio
Share of yield that is marketable.

Purpose:
Shows production quality, not just total yield.

## Season Yield
Season yield measured in kg/m2.

Purpose:
Shows production output at plot level.
'@ | Set-Content docs/kpi_definitions.md
