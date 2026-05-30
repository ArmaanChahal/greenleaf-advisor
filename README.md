# GreenLeaf CEA — Precision Agriculture Intelligence App

## Run it (30 seconds)

```bash
cd greenleaf_app
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # optional, only for Ask GreenLeaf page
streamlit run app.py
```

Opens at http://localhost:8501

## What it does

Four pages, one story:

### 1. Overview — "Does precision agriculture actually pay back?"
KPI tiles, ROI by treatment, precision benefit by treatment. Sidebar filters for Crop and Treatment that update everything instantly. The honest-limits banner at the bottom.

### 2. Triage — "Where should the crew go first this morning?"
Ranked plot list by composite urgency score (normalized, weighted: 40% stress / 25% no-response / 20% alerts / 15% delay). Click any plot to see its full stress + alert timeline.

### 3. Forecast (ARIMA) — "Which plots will be in trouble next?"
ARIMA(2,1,1) time-series models fitted to each plot's daily stress index. 7-day forecast with 95% confidence bands. Ranks plots by current stress level or rising trend. Backtest shows 4.06x predictive lift of the risk score over baseline.

### 4. Ask GreenLeaf (AI) — "Natural language Q&A"
Type a question about the data. The system builds context from the dataset summary and sends to OpenAI's GPT-4o-mini. Answers cite specific numbers from the data. Requires an OpenAI API key (set OPENAI_API_KEY env var or paste in the UI).

## The innovation angle for the hackathon

Most teams will build a Tableau dashboard. This app adds three capabilities Tableau can't deliver:

1. **ARIMA forecasting with confidence bands** — real statistical modeling, not just a weighted index. The AR(2) component captures day-over-day stress persistence. The MA(1) captures shock response. The differencing (I=1) handles trend. This is textbook time-series analysis applied to a real agricultural problem.

2. **LLM-powered natural language Q&A** — a farmer types "should I invest in High Light treatment for strawberries" and gets a plain-English answer grounded in the actual data with specific dollar amounts and confidence intervals.

3. **Interactive what-if filtering** — sidebar filters propagate instantly to every chart on every page. Click any plot in the Triage view to see its full history. The interactivity is native, not bolted on.

## Data

All 9 CSVs from the GreenLeaf CEA dataset are in the `data/` folder. The app loads them at startup and caches them for the session.

## Key numbers

- $47,033 total precision benefit across 120 plots
- Strawberry High Light: +$271/plot lift vs Control (p < 0.05)
- Strawberry Shade: -$290/plot loss vs Control (p < 0.05)
- 87% vs 59% same-day vs 2-day response improvement rate
- 4.06x predictive lift of risk score over baseline
- 1,108 worsening events out of 4,820 actions (23%)
