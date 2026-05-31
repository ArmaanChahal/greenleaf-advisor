# GreenLeaf Advisor System Workflow

## Project Name

GreenLeaf Advisor - AI-Ready Precision Agriculture Decision System

## Purpose

GreenLeaf Advisor converts greenhouse sensor, response, cost, and season outcome data into decision-ready recommendations.

The system combines:
- Tableau dashboards
- Python data preparation
- structured JSON outputs
- LLM advisor prompt rules
- farmer and lender recommendation summaries

## Workflow

1. CSV datasets are collected.
2. Python loads and cleans the CSV files.
3. Python calculates verified KPIs.
4. KPI outputs are saved as structured JSON.
5. Tableau dashboards visualize the same business logic.
6. The advisor prompt uses only verified JSON values.
7. The advisor produces farmer recommendations and lender-ready summaries.

## CSV Inputs

Main input files:
- daily_sensor_readings.csv
- daily_input_costs.csv
- input_applications.csv
- season_summary.csv
- farms.csv
- plots.csv
- greenhouses.csv

## Python Layer

Script:
scripts/generate_advisor_outputs.py

Main functions:
- rank_priority_plots
- compare_response_timing
- summarize_precision_roi
- generate_financing_memo_summary

Outputs:
- plot_priority_summary.json
- response_timing_summary.json
- precision_roi_summary.json
- financing_memo_summary.json
- advisor_recommendations.md

## Tableau Layer

Dashboard 1:
Act First - Weekly Plot Prioritization

Dashboard 2:
Response Timing Playbook

Dashboard 3:
Precision Actions vs Routine Practices

Dashboard 4:
ROI / Season Outcome Dashboard

Dashboard 5:
GreenLeaf Advisor - AI-Ready Decision System

## Advisor Layer

The advisor layer uses structured JSON outputs and strict prompt rules.

It is designed to:
- prioritize plots
- explain response timing patterns
- summarize precision cost and ROI
- generate lender-ready financing summaries

## Trust Rules

The advisor must:
- use only verified Tableau or Python-generated data
- not invent unsupported numbers
- explain recommendations with KPI evidence
- describe response timing as association, not proven causality
- return structured JSON before plain-English summaries

## Why This Matters

The system makes greenhouse decision-making easier by connecting operational alerts, response timing, input cost, and season ROI into one explainable workflow.
'@ | Set-Content docs/system_workflow.md
